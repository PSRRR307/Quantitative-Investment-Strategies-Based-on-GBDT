import pandas as pd
import numpy as np
import os


# 定义处理Volume列的函数
def convert_volume(vol_str):
    if isinstance(vol_str, str):
        unit = vol_str[-1]
        number = float(vol_str[:-1])
        if unit == 'B':
            return number * 1000000000  # 十亿
        elif unit == 'M':
            return number * 1000000     # 百万
        elif unit == 'K':
            return number * 1000        # 千
        else:
            return float(vol_str)
    else:
        return float(vol_str)


def process_data(file_path, output_file):
    """
    读取CSV文件，处理数据并保存到Excel文件中。

    参数:
    file_path (str): 输入的CSV文件路径。
    output_file (str): 输出的Excel文件路径。
    """
    # Step 1: 读取 CSV 文件
    df = pd.read_csv(file_path, encoding='utf-8')
    df['Datetime'] = df['Date']
    # # Step 2: 删除最后一列（"涨跌幅"）
    # df = df.iloc[:, :-1]

    # # Step 3: 列名映射
    # column_mapping = {
    #     "日期": "Datetime",
    #     "开盘": "Open",
    #     "高": "High",
    #     "低": "Low",
    #     "收盘": "Close",
    #     "交易量": "Volume"
    # }
    # df.rename(columns=column_mapping, inplace=True)
    if 'Volume' not in df.columns:
        print(f"Warning: 'Volume' column not found in {file_path}. Skipping this file.")
        return
    df['Volume'] = df['Volume'].apply(convert_volume)

    # Step 4: 将数值列转换为浮点数
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Step 5: 计算技术指标
    # 收益率
    df['Return'] = df['Close'].pct_change(fill_method=None)

    # 振幅
    df['Amplitude'] = (df['High'] - df['Low']) / df['High']

    # 成交量变化
    df['Volume_Change'] = df['Volume'].pct_change(fill_method=None)

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

    # Step 6: 重新排列列顺序
    columns_order = [
        'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Return', 'Amplitude', 'Volume_Change', 'MA5', 'MA10', 'MA20',
        'STD20', 'Upper_Band', 'Lower_Band', 'Momentum5', 'Momentum10',
        'Volatility20', 'Trend_MA5', 'Trend_MA10', 'RSI', 'BBP'
    ]

    df = df[columns_order]

    # 填充 NaN 值为 0
    df.fillna(0, inplace=True)

    # Step 7: 写入 Excel 文件
    df.to_excel(output_file, index=False, sheet_name='Sheet1')

    print(f"数据已成功写入 {output_file}")


def process_folders(file_path, save_path):
    # 遍历 data 文件夹中的 CSV 文件
    data_folder = file_path
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_folder, file_name)
            output_file = os.path.join(save_path, file_name.replace('.csv', '_output.xlsx'))

            # 处理数据
            process_data(file_path, output_file)
            # print(output_file, ' finish\n')


if __name__ == '__main__':
    data_folder = 'optimal/online_data'
    save_path = "optimal/preprocessed_data"
    process_folders(data_folder, save_path)