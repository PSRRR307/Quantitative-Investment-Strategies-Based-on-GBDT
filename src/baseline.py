import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

class StockPricePredictor:
    def __init__(self, task_type='regression', prediction_horizon=1):
        self.task_type = task_type
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def read_data_file(self, file_path):
        """读取数据文件，支持多种格式"""
        try:
            if file_path.endswith('.xlsx'):
                return pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            else:
                # 尝试自动检测格式
                try:
                    return pd.read_excel(file_path)
                except:
                    return pd.read_csv(file_path)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return None
        
    def prepare_features_target(self, data):
        """准备特征和目标变量"""
        df = data.copy()
        
        # 技术指标特征
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Return', 'Amplitude', 'Volume_Change',
            'MA5', 'MA10', 'MA20', 'STD20', 'Upper_Band', 'Lower_Band',
            'Momentum5', 'Momentum10', 'Volatility20', 'Trend_MA5', 'Trend_MA10',
            'RSI', 'BBP'
        ]
        
        # 确保所有特征列都存在
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"使用的特征: {len(available_features)}个")
        
        # 准备目标变量
        if self.task_type == 'regression':
            df['target'] = df['Close'].shift(-self.prediction_horizon)
        else:
            df['future_close'] = df['Close'].shift(-self.prediction_horizon)
            df['target'] = (df['future_close'] > df['Close']).astype(int)
        
        # 删除包含NaN的行
        df = df.dropna()
        
        return df, available_features
    
    def train_model(self, train_file, stock_name=None):
        """训练决策树模型"""
        print(f"正在读取训练数据: {train_file}")
        train_data = self.read_data_file(train_file)
        
        if train_data is None:
            print(f"无法读取训练文件: {train_file}")
            return None
            
        train_data, feature_columns = self.prepare_features_target(train_data)
        
        if len(train_data) < 10:
            print(f"警告: {stock_name} 训练数据不足，跳过处理")
            return None
        
        X_train = train_data[feature_columns]
        y_train = train_data['target']
        
        print(f"训练数据形状: {X_train.shape}")
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 创建模型
        if self.task_type == 'regression':
            model = DecisionTreeRegressor(random_state=42)
            param_grid = {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
            scoring = 'neg_mean_squared_error'
        else:
            model = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
            scoring = 'accuracy'
        
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)  # 减少交叉验证次数以加快速度
        
        print(f"正在为 {stock_name} 训练决策树模型...")
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, 
            scoring=scoring, n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        
        print(f"{stock_name} - 最佳参数: {grid_search.best_params_}")
        print(f"{stock_name} - 最佳得分: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def predict(self, test_file, prediction_file, stock_name=None):
        """进行预测并保存结果"""
        if self.model is None:
            print("错误: 请先训练模型")
            return None
        
        print(f"正在读取测试数据: {test_file}")
        test_data = self.read_data_file(test_file)
        
        if test_data is None:
            print(f"无法读取测试文件: {test_file}")
            return None
            
        test_data, feature_columns = self.prepare_features_target(test_data)
        
        if len(test_data) == 0:
            print(f"警告: {stock_name} 测试数据为空，跳过预测")
            return None
        
        X_test = test_data[feature_columns]
        y_test = test_data['target']
        
        # 标准化特征
        X_test_scaled = self.scaler.transform(X_test)
        
        # 进行预测
        predictions = self.model.predict(X_test_scaled)
        
        # 评估模型
        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            print(f"{stock_name} - 测试集RMSE: {rmse:.4f}")
            
            # 计算方向准确率
            actual_direction = (y_test > test_data['Close']).astype(int)
            pred_direction = (predictions > test_data['Close']).astype(int)
            direction_accuracy = accuracy_score(actual_direction, pred_direction)
            print(f"{stock_name} - 方向准确率: {direction_accuracy:.4f}")
        else:
            accuracy = accuracy_score(y_test, predictions)
            print(f"{stock_name} - 测试集准确率: {accuracy:.4f}")
        
        # 保存预测结果
        result_df = test_data.copy()
        result_df['Predictions'] = predictions
        result_df['Actual'] = y_test.values
        
        # 保存到文件
        try:
            if prediction_file.endswith('.xlsx'):
                result_df.to_excel(prediction_file, index=False)
            else:
                result_df.to_csv(prediction_file, index=False)
            print(f"{stock_name} 预测结果已保存到: {prediction_file}")
        except Exception as e:
            print(f"保存结果失败: {e}")
            # 尝试保存为CSV格式
            csv_file = prediction_file.replace('.xlsx', '.csv')
            result_df.to_csv(csv_file, index=False)
            print(f"{stock_name} 预测结果已保存到: {csv_file}")
        
        return result_df

def run_simple_stock_prediction():
    """简化的股票价格预测"""
    
    # 配置路径
    training_folder = "optimal/training_set"
    test_folder = "optimal/test_set"
    prediction_folder = "optimal/prediction_dt"
    
    # 创建预测文件夹
    os.makedirs(prediction_folder, exist_ok=True)
    
    # 配置预测参数
    task_type = 'regression'
    prediction_horizon = 1
    
    all_results = []
    processed_count = 0
    
    # 遍历训练集文件
    for file_name in os.listdir(training_folder):
        if file_name.endswith('_train_data.xlsx') or file_name.endswith('_train_data.csv'):
            stock_name = file_name.split('_')[0]
            print(f"\n{'='*50}")
            print(f"处理股票: {stock_name}")
            print(f"{'='*50}")
            
            # 文件路径
            train_file = os.path.join(training_folder, file_name)
            test_file_name = file_name.replace('_train_data.', '_test_data.')
            test_file = os.path.join(test_folder, test_file_name)
            prediction_file = os.path.join(prediction_folder, file_name.replace('_train_data.', '_predictions.'))
            
            # 检查测试文件是否存在
            if not os.path.exists(test_file):
                print(f"测试文件不存在: {test_file}")
                continue
            
            try:
                # 创建预测器
                predictor = StockPricePredictor(
                    task_type=task_type, 
                    prediction_horizon=prediction_horizon
                )
                
                # 训练模型
                model = predictor.train_model(train_file, stock_name)
                
                if model is not None:
                    # 进行预测
                    results = predictor.predict(test_file, prediction_file, stock_name)
                    
                    if results is not None:
                        # 计算评估指标
                        rmse = np.sqrt(mean_squared_error(results['Actual'], results['Predictions']))
                        direction_accuracy = accuracy_score(
                            (results['Actual'] > results['Close']).astype(int),
                            (results['Predictions'] > results['Close']).astype(int)
                        )
                        all_results.append({
                            'Stock': stock_name,
                            'RMSE': rmse,
                            'Direction_Accuracy': direction_accuracy,
                            'Samples': len(results)
                        })
                        processed_count += 1
                        
                        # 显示特征重要性
                        print(f"\n{stock_name} 前3个重要特征:")
                        sorted_features = sorted(predictor.feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True)
                        for feature, importance in sorted_features[:3]:
                            print(f"  {feature}: {importance:.4f}")
                
            except Exception as e:
                print(f"处理 {stock_name} 时出现错误: {str(e)}")
                continue
    
    # 分析总体结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*50}")
        print("总体结果统计:")
        print(f"{'='*50}")
        print(f"成功处理股票数量: {processed_count}")
        print(f"平均RMSE: {results_df['RMSE'].mean():.4f}")
        print(f"平均方向准确率: {results_df['Direction_Accuracy'].mean():.4f}")
        
        # 保存总体结果
        try:
            results_df.to_excel(os.path.join(prediction_folder, 'overall_results.xlsx'), index=False)
        except:
            results_df.to_csv(os.path.join(prediction_folder, 'overall_results.csv'), index=False)
        
        return results_df
    return None

if __name__ == '__main__':
    print("开始股票价格预测...")
    
    # 检查必要的库
    try:
        import pandas as pd
        from sklearn.tree import DecisionTreeRegressor
        print("必要的库已安装")
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请运行: pip install pandas scikit-learn matplotlib seaborn openpyxl")
        exit(1)
    
    # 运行股票预测
    results = run_simple_stock_prediction()
    
    if results is not None:
        print(f"\n预测完成! 共处理 {len(results)} 个股票")
        print("预测结果保存在: optimal/prediction_dt/")
    else:
        print("没有成功处理任何股票数据")