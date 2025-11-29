import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 定义辅助函数和策略类
def raw_prediction_to_signal(pred: pd.Series, total_cash: float, long_only: bool = False) -> pd.Series:
    if len(pred) > 1:
        if not long_only:
            pred -= pred.groupby(level=0).mean()
        pred_ = pred.copy()
        abs_sum = abs(pred_).groupby(level=0).sum()
        abs_sum = abs_sum.replace(0, 1e-10)  # 避免除以零错误
        pred_ /= abs_sum
        return pred_ * total_cash
    else:
        return pred * total_cash

# 计算交易数额
def get_trade_volume(signal: pd.Series, price: pd.Series) -> pd.Series:
    trade_volume = abs(signal) / price  # 单位是股
    return (trade_volume + 0.5).astype(int)  # 四舍五入取整

# 获取开盘价
def get_price(data: pd.DataFrame, price: str = "Open_price") -> dict:
    if price not in data.columns:
        raise ValueError(f"Column '{price}' not found in DataFrame")
    current_price = data.set_index("Stock")[price].to_dict()
    return {stock: max(value, 1e-10) for stock, value in current_price.items()}  # 确保价格有效

# 获取交易量
def get_vol(data: pd.DataFrame, volume: str = "trade_volume") -> dict:
    if volume not in data.columns:
        raise ValueError(f"Column '{volume}' not found in DataFrame")
    current_volume = data.set_index("Stock")[volume].to_dict()
    return current_volume

# 选出正收益的股票
def check_signal(order: dict) -> dict:
    return {k: v for k, v in order.items() if v > 0}

# 带手续费的每日收益
def calculate_daily_profit(order: dict, price: dict, position: dict, cost: dict,
                           cash_available: float, commission_rate: float = 0.001) -> float:
    daily_profit = 0.0
    # 计算卖出收益
    for code, volume in order["sell"].items():
        if code in position:
            sell_price = price.get(code, 0)
            if sell_price <= 0:
                continue
            buy_price = cost.get(code, 0)
            # 计算手续费
            commission = sell_price * volume * commission_rate
            # 计算利润（扣除手续费）
            daily_profit += (sell_price - buy_price) * volume - commission
            # 更新可用现金（扣除手续费）
            cash_available += sell_price * volume - commission
            # 更新持仓
            position[code] -= volume
            if position[code] <= 0:
                del position[code]
                del cost[code]

    # 处理买入交易
    for code, volume in order["buy"].items():
        buy_price = price.get(code, 0)
        if buy_price <= 0:
            continue
        # 计算手续费
        commission = buy_price * volume * commission_rate
        # 更新可用现金（扣除买入金额和手续费）
        cash_available -= buy_price * volume + commission
        # 更新持仓和成本
        if code in position:
            total_cost = cost[code] * position[code] + buy_price * volume
            total_volume = position[code] + volume
            cost[code] = total_cost / total_volume
        else:
            cost[code] = buy_price
        position[code] = position.get(code, 0) + volume

    return daily_profit

# 定义了基础策略类
class BaseStrategy:
    def __init__(self, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.k = kwargs.get("k", 0.2)                           # 选股比例
        self.unit = kwargs.get("unit", "lot")                   # 交易单位（手数）
        self.risk_degree = kwargs.get("risk_degree", 0.95)      # 风险控制参数
        self.max_volume = kwargs.get("max_volume", 0.05)        # 最大交易量限制
        self.long_only = kwargs.get("long_only", False)         # 是否只做多

# 继承了基础策略类，用QlibTopK策略选择排名前 K 的股票
class QlibTopKStrategy(BaseStrategy):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        self.n_start = kwargs.get("n_start", None)              # 初始选股数量
        self.equal_weight = kwargs.get("equal_weight", False)   # 是否等权重分配资金
        self.long_only = True                                   # 只做多

    # 生成交易信号
    def to_signal(self, data: pd.DataFrame, position: dict, cash_available: float = None):
        if cash_available is None or cash_available <= 0:
            raise ValueError("cash_available must be a positive value")

        n_k = int(len(data) * self.k + 0.5) or 1                # 四舍五入计算选股数(至少一只)
        valid_position = check_signal(position)                 # 返回有效的持仓字典
        price = get_price(data, price="Open_price")             # 获取股票的开盘价(用于计算交易量)
        # 计算预计收益率
        data["return_ratio"] = (data["Predict_close_price"] - data["Open_price"]) / data["Open_price"]
        # 如果没有持仓，初始化买入
        if len(valid_position) == 0:
            if self.n_start is not None:
                n_k = self.n_start
            data_buy = data.sort_values("return_ratio", ascending=False).head(n_k)
            sell_dict = {}
        # 如果有持仓，计算需要调仓的股票数量（swap_k）
        else:
            swap_k = int(len(valid_position) * self.k + 0.5) or 1
            data_in_position = data[data["Stock"].isin(valid_position.keys())]
            data_sell = data_in_position[data_in_position["return_ratio"] < 0]
            sell_dict = {k: v for k, v in valid_position.items() if k in data_sell["Stock"].values}
            data_not_in_position = data[~data["Stock"].isin(valid_position.keys())]
            data_buy = data_not_in_position.sort_values("return_ratio", ascending=False).head(swap_k)

        max_investment = cash_available * self.max_volume       # 最大投资金额
        risk_adjusted_investment = max_investment * self.risk_degree  # 风险调整后的投资金额

        if self.equal_weight:
            # 等权重分配，调整每个股票的投资金额
            investment_per_stock = risk_adjusted_investment / len(data_buy)
            data_buy["predict"] = investment_per_stock
        else:
            # 以预计收益比率（预估收益 / 开盘价格）为权重，调整投资金额
            data_buy["expected_return_ratio"] = data_buy["Predict_close_price"] / data_buy["Open_price"]
            sum_return_ratio = data_buy["expected_return_ratio"].sum()
            if sum_return_ratio == 0:
                # 避免除以零错误，等权重分配
                investment_per_stock = risk_adjusted_investment / len(data_buy)
                data_buy["predict"] = investment_per_stock
            else:
                data_buy["predict"] = data_buy["expected_return_ratio"] / sum_return_ratio * risk_adjusted_investment

        data_buy["trade_volume"] = get_trade_volume(data_buy["predict"], data_buy["Open_price"])
        buy_dict = check_signal(get_vol(data_buy, volume="trade_volume"))
        sell_dict = check_signal(sell_dict)

        return {"buy": buy_dict, "sell": sell_dict}, price

def process_folder(folder_path: str, strategy: BaseStrategy, initial_cash: float = 100000.0):
    files = os.listdir(folder_path)
    excel_files = [f for f in files if f.endswith('.xlsx') or f.endswith('.csv')]
    position = {}
    cost = {}
    cash_available = initial_cash
    daily_profits = []
    cumulative_profits = []
    max_drawdown = 0.0  # 最大回撤
    max_drawdown_cash = 0.0 # 最大回撤数额
    peak_value = initial_cash  # 峰值

    excel_files.sort()

    for file_name in excel_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_name}")

        try:
            data = pd.read_csv(file_path) if file_name.endswith('.csv') else pd.read_excel(file_path)

            required_columns = ["Stock", "Predict_close_price", "Open_price"]
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"File {file_name} is missing required columns: {required_columns}")

            order, price = strategy.to_signal(data, position, cash_available)

            daily_profit = calculate_daily_profit(order, price, position, cost, cash_available)
            cash_available += daily_profit
            daily_profits.append(daily_profit)

            cumulative_profit = sum(daily_profits)
            cumulative_profits.append(cumulative_profit)

            # 计算最大回撤
            current_value = initial_cash + cumulative_profit  # 当前总资产
            if current_value > peak_value:
                peak_value = current_value  # 更新峰值
            drawdown_cash = peak_value - current_value  # 回撤数额
            drawdown = (drawdown_cash) / peak_value  # 当前回撤
            if drawdown > max_drawdown:
                max_drawdown = drawdown  # 更新最大回撤
            if drawdown_cash > max_drawdown_cash:
                max_drawdown_cash = drawdown_cash

            print(f"当天利润: {daily_profit}, 累计利润: {cumulative_profit}, 收益比: {(cash_available-initial_cash) / initial_cash*100:.4f}%")
            print(f"现金余额: {cash_available}, 持仓: {position}")
            print(f"当前回撤: {drawdown:.4f}, 最大回撤: {max_drawdown:.4f},最大回撤数额: {max_drawdown_cash:.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # 绘制累计利润曲线
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_profits, label="Cumulative Profit")
    plt.xlabel("Days")
    plt.ylabel("Profit")
    plt.title(f"Strategy Profit (Max Drawdown: {max_drawdown:.4f})")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # 示例调用
    folder_path = "output_files"
    my_kwargs = {
        'k': 0.2,
        'unit': 'lot',
        'risk_degree': 0.95,
        'max_volume': 0.10,
    }
    strategy = QlibTopKStrategy(my_kwargs)
    initial_cash = 100000.0
    process_folder(folder_path, strategy, initial_cash)