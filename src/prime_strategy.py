import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def prime_strategy(predictions_file, test_data_file, initial_capital=100000, transaction_cost=0.001):
    """
    运行策略并返回投资组合价值、总收益、收益率和最大回撤。

    参数:
    predictions_file (str): 预测数据的文件路径。
    test_data_file (str): 真实数据的文件路径。
    initial_capital (float): 初始资金，默认为100000。
    transaction_cost (float): 交易成本，默认为0.001（0.1%）。

    返回:
    portfolio_df (DataFrame): 每日投资组合价值的DataFrame。
    total_return (float): 总收益。
    return_rate (float): 收益率（百分比）。
    max_drawdown (float): 最大回撤（百分比）。
    """
    # 读取预测数据和真实数据
    predictions_df = pd.read_excel(predictions_file)  # 预测数据
    testset_df = pd.read_excel(test_data_file)  # 真实数据

    # 确保日期列是 datetime 类型
    predictions_df['Datetime'] = pd.to_datetime(predictions_df['Datetime'])
    testset_df['Datetime'] = pd.to_datetime(testset_df['Datetime'])

    # 合并预测数据和真实数据
    merged_df = pd.merge(predictions_df, testset_df, on='Datetime', suffixes=('_pred', '_true'))

    # 初始化策略参数
    cash = initial_capital  # 当前现金
    position = 0  # 当前持仓
    portfolio_value = []  # 每日投资组合价值
    max_drawdown = 0  # 最大回撤

    # 策略逻辑
    for i in range(len(merged_df)):
        row = merged_df.iloc[i]
        date = row['Datetime']
        open_true = row['Open_true']
        close_true = row['Close_true']
        open_pred = row['Open_pred']
        close_pred = row['Close_pred']
        low_pred = row['Low_pred']

        # 计算最大回撤
        if i > 0 and position > 0:
            peak_value = max([pv['Value'] for pv in portfolio_value])
            current_drawdown = (peak_value - daily_value) / peak_value
            max_drawdown = max(max_drawdown, current_drawdown)

        # 买入条件：预测收盘价高于真实开盘价，且当前没有持仓
        if close_pred > open_true and position == 0:
            # 买入全部资金对应的股票
            position = cash / open_true
            cash = 0
            position *= (1 - transaction_cost)  # 扣除交易成本

        # 卖出条件：预测收盘价低于真实开盘价，且当前有持仓
        elif close_pred < open_true and position > 0:
            # 卖出全部持仓
            cash = position * close_true
            position = 0
            cash *= (1 - transaction_cost)  # 扣除交易成本

        # 计算每日投资组合价值
        daily_value = cash + position * close_true
        portfolio_value.append({'Datetime': date, 'Value': daily_value})

    # 将投资组合价值转换为 DataFrame
    portfolio_df = pd.DataFrame(portfolio_value)

    # 计算总收益和收益率
    final_value = portfolio_df['Value'].iloc[-1]
    total_return = final_value - initial_capital
    return_rate = (final_value - initial_capital) / initial_capital * 100

    return portfolio_df, total_return, return_rate, max_drawdown


def plot_portfolio_value(portfolio_df):
    """
    绘制投资组合价值随时间的变化图。

    参数:
    portfolio_df (DataFrame): 包含日期和投资组合价值的DataFrame。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_df['Datetime'], portfolio_df['Value'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()


# 调用
def run_strategy(predictions_file, test_data_file, initial_capital, transaction_cost):
    portfolio_df, total_return, return_rate, max_drawdown = \
        prime_strategy(predictions_file, test_data_file, initial_capital, transaction_cost)

    # 打印结果
    print(f"初始资金: {initial_capital}")
    print(f"最终资金: {portfolio_df['Value'].iloc[-1]:.2f}")
    print(f"总收益: {total_return:.2f}")
    print(f"收益率: {return_rate:.2f}%")
    print(f"最大回撤: {max_drawdown:.2%}")

    # 绘制投资组合价值图
    plot_portfolio_value(portfolio_df)


if __name__ == "__main__":
    predictions_file = 'daily_predictions.xlsx'
    test_data_file = 'daily_test_data.xlsx'
    initial_capital = 100000
    transaction_cost = 0.001
    run_strategy(predictions_file, test_data_file, initial_capital, transaction_cost)
