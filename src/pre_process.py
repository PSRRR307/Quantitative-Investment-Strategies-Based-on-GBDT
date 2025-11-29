import pandas as pd
import numpy as np


def process_data(file_path, output_file, data_type):
    """
    读取TXT文件，处理数据并保存到Excel文件中。

    参数:
    file_path (str): 输入的TXT文件路径。
    output_file (str): 输出的Excel文件路径。
    data_type (str): 数据类型，'minute' 或 'daily'
    """
    # Step 1: 读取 TXT 文件
    data = []
    with open(file_path, 'r', encoding='GBK') as file:
        for line in file:
            if data_type == 'minute' and line.strip() and not line.startswith("゜ヽ"):  # 跳过非数据行
                data.append(line.strip().split())
            elif data_type == 'daily':
                data.append(line.strip().split())

    # Step 2: 解析数据并构建 DataFrame
    if data_type == 'minute':
        columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    elif data_type == 'daily':
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']

    df = pd.DataFrame(data, columns=columns)

    # 将数值列转换为浮点数
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Step 3: 合并 'Date' 和 'Time' 为 'Datetime'
    if data_type == 'minute':
        df['Datetime'] = df['Date'] + ' ' + df['Time']
    elif data_type == 'daily':
        df['Datetime'] = df['Date']

    # Step 4: 计算技术指标
    # 收益率
    df['Return'] = df['Close'].pct_change(fill_method=None)

    # 振幅
    df['Amplitude'] = (df['High'] - df['Low']) / df['High']

    # 成交量变化
    # Step 1: Calculate percentage change
    df['Volume_Change'] = df['Volume'].pct_change(fill_method=None)

    # Step 2: Identify where previous Volume is zero
    prev_zero = df['Volume'].shift(1) == 0

    # Step 3: Compute maximum change excluding 'inf'
    # Replace 'inf' with NaN to exclude them from the max calculation
    temp = df['Volume_Change'].replace([np.inf, -np.inf], np.nan)
    max_change = temp.max()

    # Step 4: Replace 'inf' with the maximum change
    df.loc[prev_zero, 'Volume_Change'] = max_change

    # df['Volume_Change'] = df['Volume'].pct_change()

    # 移动平均线
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # 标准差
    df['STD20'] = df['Close'].rolling(window=20).std()

    # 布林带
    df['Upper_Band'] = df['MA20'] + 2 * df['STD20']
    df['Lower_Band'] = df['MA20'] - 2 * df['STD20']

    # 动量指标
    df['Momentum5'] = df['Close'].diff(5)
    df['Momentum10'] = df['Close'].diff(10)

    # 波动率
    df['Volatility20'] = df['Return'].rolling(window=20).std()

    # 趋势指标
    df['Trend_MA5'] = df['MA5'] - df['MA10']
    df['Trend_MA10'] = df['MA10'] - df['MA20']

    # RSI 计算
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 布林带百分比 (BBP)
    df['BBP'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])

    # Step 5: 重新排列列顺序
    if data_type == 'minute':
        columns_order = [
            'Datetime', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover',
            'Return', 'Amplitude', 'Volume_Change', 'MA5', 'MA10', 'MA20',
            'STD20', 'Upper_Band', 'Lower_Band', 'Momentum5', 'Momentum10',
            'Volatility20', 'Trend_MA5', 'Trend_MA10', 'RSI', 'BBP'
        ]
    elif data_type == 'daily':
        columns_order = [
            'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover',
            'Return', 'Amplitude', 'Volume_Change', 'MA5', 'MA10', 'MA20',
            'STD20', 'Upper_Band', 'Lower_Band', 'Momentum5', 'Momentum10',
            'Volatility20', 'Trend_MA5', 'Trend_MA10', 'RSI', 'BBP'
        ]

    df = df[columns_order]

    # 填充 NaN 值为 0
    df.fillna(0, inplace=True)

    # Step 6: 写入 Excel 文件
    df.to_excel(output_file, index=False, sheet_name='Sheet1')

    print(f"数据已成功写入 {output_file}")


if __name__ == '__main__':
    # 示例调用
    file_path = "HSTECH_5min.txt"
    output_file = "5min_Output.xlsx"
    process_data(file_path, output_file, data_type='minute')

    file_path = "HSTECH.txt"
    output_file = "daily_Output.xlsx"
    process_data(file_path, output_file, data_type='daily')