import os
import pandas as pd
from datetime import datetime, timedelta


def read_stock_data(prediction_folder, true_folder, date):
    """
    根据给定的日期，从预测文件和真实数据文件中提取数据，并返回一个包含结果的DataFrame。
    """
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=['Stock', 'Predict_close_price', 'Open_price'])

    # Iterate over each file in the prediction folder
    for file in os.listdir(prediction_folder):
        if file.endswith('_predictions.xlsx'):
            stock_name = file.split('_')[0]  # Get stock name from file name

            # Read the prediction file
            prediction_path = os.path.join(prediction_folder, file)
            prediction_df = pd.read_excel(prediction_path)

            # Ensure that the 'Date' column is in datetime format for comparison
            prediction_df['Datetime'] = pd.to_datetime(prediction_df['Datetime'])

            # Try to filter the prediction data for the specific date
            prediction_row = prediction_df[prediction_df['Datetime'] == date]
            if prediction_row.empty:
                print(f"Warning: {stock_name} does not have prediction data for {date}. Skipping...")
                continue
            prediction_close = prediction_row['Close'].values[0]

            # Read the corresponding true file
            true_path = os.path.join(true_folder, f"{stock_name}_test_data.xlsx")
            true_df = pd.read_excel(true_path)

            # Ensure that the 'Date' column in true data is also in datetime format
            true_df['Datetime'] = pd.to_datetime(true_df['Datetime'])

            # Try to filter the true data for the specific date
            true_row = true_df[true_df['Datetime'] == date]
            if true_row.empty:
                print(f"Warning: {stock_name} does not have true data for {date}. Skipping...")
                continue
            true_open = true_row['Open'].values[0]

            # Create a temporary DataFrame for the current stock's data
            temp_df = pd.DataFrame({
                'Stock': [stock_name],
                'Predict_close_price': [prediction_close],
                'Open_price': [true_open]
            })

            # Concatenate the temporary DataFrame with the result DataFrame
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

    return result_df


def combination(prediction_folder, true_folder, output_folder):
    # 设定起始日期和结束日期
    start_date = datetime(2024, 1, 3)  # 起始日期
    end_date = datetime(2024, 9, 30)  # 结束日期

    # 生成交易日
    trading_days = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')

    # 输出文件夹路径
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有交易日期并生成文件
    for date_str in trading_days:
        # 转换日期为datetime格式
        specific_date = datetime.strptime(date_str, '%Y-%m-%d')

        # 读取该日期的股票数据
        result_dataframe = read_stock_data(prediction_folder, true_folder, specific_date)

        # 生成文件名，命名规则为交易日期
        filename = os.path.join(output_folder, f'result_{specific_date.strftime("%Y-%m-%d")}.csv')

        # 保存结果到CSV文件
        result_dataframe.to_csv(filename, index=False)

        print(f'File saved: {filename}')


if __name__ == '__main__':
    # 设置预测数据和真实数据的文件夹路径
    prediction_folder = 'optimal/prediction'
    true_folder = 'optimal/test_set'

    # 输出文件夹路径
    output_folder = 'output_files'

    combination(prediction_folder, true_folder, output_folder)