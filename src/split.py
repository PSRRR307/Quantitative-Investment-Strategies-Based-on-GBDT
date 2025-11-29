import pandas as pd


def split_data(input_file, train_output_file, test_output_file):
    # 读取Excel文件，header=0表示第一行是标题行
    df = pd.read_excel(input_file, header=0)

    # 删除前20行数据（不包含标题行）
    df = df.drop(df.index[:20])

    # 确保Datetime列是字符串形式
    df['Datetime'] = df['Datetime'].astype(str)

    # 提取Datetime列的前两位
    df['Prefix'] = df['Datetime'].str[:4]

    # 划分训练集和测试集
    train_df = df[df['Prefix'].isin(['2022', '2023'])]
    test_df = df[df['Prefix'] == '2024']

    # 输出到两个Excel文件，保留标题行
    train_df.to_excel(train_output_file, index=False, header=True)
    test_df.to_excel(test_output_file, index=False, header=True)
    print("split data done")


if __name__ == '__main__':
    # 示例调用
    input_file1 = 'daily_Output.xlsx'
    train_output_file1 = 'daily_train_data.xlsx'
    test_output_file1 = 'daily_test_data.xlsx'

    split_data(input_file1, train_output_file1, test_output_file1)

    # 示例调用
    input_file2 = '5min_Output.xlsx'
    train_output_file2 = '5min_train_data.xlsx'
    test_output_file2 = '5min_test_data.xlsx'

    split_data(input_file2, train_output_file2, test_output_file2)