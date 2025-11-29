from pre_process import process_data
from split import split_data
from GBDT import gbdt_main
from prime_strategy import run_strategy

origin_daily_file = "origin/HSTECH.txt"
origin_5min_file = "origin/HSTECH_5min.txt"

processed_daily_file = "preprocessed_data/daily.xlsx"
processed_5min_file = "preprocessed_data/5min.xlsx"

training_daily_file = "training_set/daily.xlsx"
training_5min_file = "training_set/5min.xlsx"
test_daily_file = "test_set/daily.xlsx"
test_5min_file = "test_set/5min.xlsx"

prediction_daily_file = "prediction/daily.xlsx"
prediction_5min_file = "prediction/5min.xlsx"

initial_capital = 100000
transaction_cost = 0.001


def main():
    # # 数据预处理，计算因子，并输出为excel文件
    # process_data(origin_daily_file, processed_daily_file, 'daily')
    # # process_data(origin_5min_file, processed_5min_file, 'minute')
    #
    # # 生成训练集和测试集
    # split_data(processed_daily_file, training_daily_file, test_daily_file)
    # # split_data(processed_5min_file, training_5min_file, test_5min_file)
    #
    # # 训练模型并生成预测值
    # gbdt_main(training_daily_file, test_daily_file, prediction_daily_file)
    # # gbdt_main(training_5min_file, test_5min_file, prediction_5min_file)

    # 使用基础策略模型
    run_strategy(prediction_daily_file, test_daily_file, initial_capital, transaction_cost)
    # run_strategy(prediction_5min_file, test_5min_file, initial_capital, transaction_cost)


if __name__ == '__main__':
    main()
