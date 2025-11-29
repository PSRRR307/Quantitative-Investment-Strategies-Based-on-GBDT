from read import process_folders
from split import split_data
from GBDT import gbdt_main
import os
from data_combination import combination
from best_stategy import process_folder, QlibTopKStrategy


data_folder = 'optimal/online_data'
processed_folder = "optimal/preprocessed_data"
training_folder = "optimal/training_set"
test_folder = "optimal/test_set"
prediction_folder = "optimal/prediction"

# 输出文件夹路径
output_folder = 'output_files'
my_kwargs = {
    'k': 0.2,
    'unit': 'lot',
    'risk_degree': 0.95,
    'max_volume': 0.05,
}
strategy = QlibTopKStrategy(my_kwargs)
initial_cash = 100000


def main():
    # 数据预处理，计算因子，并输出为 Excel 文件
    process_folders(data_folder, processed_folder)

    # 遍历处理后的文件夹
    for file_name in os.listdir(processed_folder):
        if file_name.endswith('.xlsx'):
            stock_name = file_name.split('_')[0]
            input_file = os.path.join(processed_folder, file_name)

            # 生成训练集和测试集文件路径
            train_output_file = os.path.join(training_folder, file_name.replace('_output.xlsx', '_train_data.xlsx'))
            test_output_file = os.path.join(test_folder, file_name.replace('_output.xlsx', '_test_data.xlsx'))

            # 划分训练集和测试集
            split_data(input_file, train_output_file, test_output_file)

            # 生成预测结果文件路径
            prediction_file = os.path.join(prediction_folder, file_name.replace('_output.xlsx', '_predictions.xlsx'))

            # 训练模型并生成预测值
            gbdt_main(train_output_file, test_output_file, prediction_file,stock_name)

    # 运行策略
    combination(prediction_folder, test_folder, output_folder)
    process_folder(output_folder, strategy, initial_cash)


if __name__ == '__main__':
    main()
