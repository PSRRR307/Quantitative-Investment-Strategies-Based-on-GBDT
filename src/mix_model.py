import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
np.random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories if they don't exist
os.makedirs("picture", exist_ok=True)
os.makedirs("model", exist_ok=True)


# Load the dataset
file_path = "training_set/daily.xlsx"
training_data = pd.read_excel(file_path)
test_file_path = "test_set/daily.xlsx"
test_data = pd.read_excel(test_file_path)

# Assuming the dataset has 'Date', 'Open', 'High', 'Low', 'Close' columns
training_data['Datetime'] = pd.to_datetime(training_data['Datetime'])
training_data.set_index('Datetime', inplace=True)

test_data['Datetime'] = pd.to_datetime(test_data['Datetime'])
test_data.set_index('Datetime', inplace=True)

# Create lagged features for training data
def create_lagged_features(data, lag_num=1):
    lagged_data = data.copy()
    for k in range(1, lag_num + 1):
        lagged_data[f'Open_Lag_{k}'] = lagged_data['Open'].shift(k)
        lagged_data[f'High_Lag_{k}'] = lagged_data['High'].shift(k)
        lagged_data[f'Low_Lag_{k}'] = lagged_data['Low'].shift(k)
        lagged_data[f'Close_Lag_{k}'] = lagged_data['Close'].shift(k)
    return lagged_data


# Create lagged features for training and test data
lag = 1  # You can change this to create more lags
training_lagged = create_lagged_features(training_data, lag)
test_lagged = create_lagged_features(test_data, lag)

# Drop rows with NaN values (due to shifting)
training_lagged.dropna(inplace=True)
test_lagged.dropna(inplace=True)

# Define features and target variables (only for Close column)
X_train = training_lagged
y_train = training_lagged['Close']

X_test = test_lagged
y_test = test_lagged['Close']

# # Define the parameter grid for hyperparameter tuning
# param_grid_rf = {
#     'n_estimators': [50, 100, 200],  # 树的数量
#     'max_depth': [None, 10, 20, 30],  # 树的最大深度，None 表示不限制
#     'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
#     'min_samples_leaf': [1, 2, 4],  # 叶节点所需的最小样本数
#     'bootstrap': [True, False]  # 是否使用有放回抽样
# }

# param_grid_xgb = {
#     'n_estimators': [50, 100, 200],  # 树的数量
#     'learning_rate': [0.01, 0.1, 0.2],  # 学习率
#     'max_depth': [3, 5, 7],  # 树的最大深度
#     'subsample': [0.8, 0.9, 1.0],  # 每棵树使用的样本比例
#     'colsample_bytree': [0.8, 0.9, 1.0],  # 每棵树使用的特征比例
#     'gamma': [0, 0.1, 0.2],  # 分裂所需的最小损失减少
#     'reg_alpha': [0, 0.1, 1],  # L1 正则化项
#     'reg_lambda': [0, 0.1, 1]  # L2 正则化项
# }

# param_grid_catboost = {
#     'iterations': [100, 200, 300],  # 树的数量
#     'learning_rate': [0.01, 0.1, 0.2],  # 学习率
#     'depth': [4, 6, 8],  # 树的最大深度
#     'l2_leaf_reg': [1, 3, 5],  # L2 正则化项
#     'border_count': [32, 64, 128],  # 特征分桶的数量
#     'subsample': [0.8, 0.9, 1.0],  # 每棵树使用的样本比例
#     'random_strength': [0, 0.1, 1]  # 随机强度，控制分裂的随机性
# }
param_grid_gbdt = {
    'n_estimators': [50],
    'learning_rate': [0.1],
    'max_depth': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [5]
}
param_grid_rf = {
    'n_estimators': [100],  # 树的数量
    'max_depth': [None],  # 树的最大深度，None 表示不限制
    'min_samples_split': [2],  # 分裂内部节点所需的最小样本数
    'min_samples_leaf': [4],  # 叶节点所需的最小样本数
}
param_grid_xgb = {
    'n_estimators': [100],  # 树的数量
    'learning_rate': [0.1],  # 学习率
    'max_depth': [3],  # 树的最大深度
    'subsample': [1.0],  # 每棵树使用的样本比例
    'colsample_bytree': [0.8],  # 每棵树使用的特征比例
    'gamma': [0.1],  # 分裂所需的最小损失减少
    'reg_alpha': [0,0.001],  # L1 正则化项
    'reg_lambda': [0,0.001]  # L2 正则化项
}
param_grid_catboost = {
    'iterations': [200],  # 树的数量
    'learning_rate': [0.05],  # 学习率
    'depth': [6],  # 树的最大深度
    'l2_leaf_reg': [3],  # L2 正则化项
    'border_count': [64],  # 特征分桶的数量
    'subsample': [1.0],  # 每棵树使用的样本比例
    'random_strength': [0.1]  # 随机强度，控制分裂的随机性
}


# Create models
gbdt_model = GradientBoostingRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
catboost_model = CatBoostRegressor(random_state=42, verbose=0)

# Set up TimeSeriesSplit for rolling window cross-validation
tscv = TimeSeriesSplit(n_splits=5, gap=20, test_size=20)

# Set up GridSearchCV for hyperparameter tuning with cross-validation
grid_search_gbdt = GridSearchCV(estimator=gbdt_model,
                           param_grid=param_grid_gbdt,
                           scoring='neg_root_mean_squared_error',
                           cv=tscv,
                           verbose=1)
grid_search_rf = GridSearchCV(estimator=rf_model,
                              param_grid=param_grid_rf,
                              scoring='neg_root_mean_squared_error',
                              cv=tscv,
                              verbose=1)

grid_search_xgb = GridSearchCV(estimator=xgb_model,
                               param_grid=param_grid_xgb,
                               scoring='neg_root_mean_squared_error',
                               cv=tscv,
                               verbose=1,
                               error_score='raise')

grid_search_catboost = GridSearchCV(estimator=catboost_model,
                                    param_grid=param_grid_catboost,
                                    scoring='neg_root_mean_squared_error',
                                    cv=tscv,
                                    verbose=1)

# Fit GridSearchCV on the entire training set for Close column
target = 'Close'

# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Fit GBDT model
grid_search_gbdt.fit(X_train_split, y_train_split)
best_params_gbdt = grid_search_gbdt.best_params_
best_score_gbdt = -grid_search_gbdt.best_score_

print(f'Best Parameters for GBDT {target}: {best_params_gbdt}')
print(f'Best Cross-Validated RMSE for GBDT {target}: {best_score_gbdt:.4f}')

best_model_gbdt = GradientBoostingRegressor(**best_params_gbdt)
best_model_gbdt.fit(X_train, y_train)


# Fit Random Forest model
grid_search_rf.fit(X_train_split, y_train_split)
best_params_rf = grid_search_rf.best_params_
best_score_rf = -grid_search_rf.best_score_

print(f'Best Parameters for Random Forest {target}: {best_params_rf}')
print(f'Best Cross-Validated RMSE for Random Forest {target}: {best_score_rf:.4f}')

best_model_rf = RandomForestRegressor(**best_params_rf)
best_model_rf.fit(X_train, y_train)

# Fit XGBoost model
grid_search_xgb.fit(X_train_split, y_train_split)
best_params_xgb = grid_search_xgb.best_params_
best_score_xgb = -grid_search_xgb.best_score_

print(f'Best Parameters for XGBoost {target}: {best_params_xgb}')
print(f'Best Cross-Validated RMSE for XGBoost {target}: {best_score_xgb:.4f}')

best_model_xgb = XGBRegressor(**best_params_xgb)
best_model_xgb.fit(X_train, y_train)

# Fit CatBoost model
grid_search_catboost.fit(X_train_split, y_train_split)
best_params_catboost = grid_search_catboost.best_params_
best_score_catboost = -grid_search_catboost.best_score_

print(f'Best Parameters for CatBoost {target}: {best_params_catboost}')
print(f'Best Cross-Validated RMSE for CatBoost {target}: {best_score_catboost:.4f}')

best_model_catboost = CatBoostRegressor(**best_params_catboost, verbose=0)
best_model_catboost.fit(X_train, y_train)


class MixModel:
    def __init__(self, *base_models):
        self.base_models = base_models
        self.meta_model = LinearRegression()

    def fit(self, X, y):
        # Get predictions from base models
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models])

        # Train the metamodel on the base models' predictions
        self.meta_model.fit(base_predictions, y)

    def predict(self, X):
        # Get predictions from base models
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models])

        # Predict using the metamodel
        return self.meta_model.predict(base_predictions)


# Create Mix model
mix_model = MixModel(best_model_gbdt,best_model_rf, best_model_xgb, best_model_catboost)
# Ensure the MixModel is fitted before making predictions
mix_model.fit(X_train, y_train)  # Ensure fit is called before predict

# Make predictions on both training and test sets using all models
models = {
    'GBDT':best_model_gbdt,
    'Random Forest': best_model_rf,
    'XGBoost': best_model_xgb,
    'CatBoost': best_model_catboost,
    'Mix': mix_model
}

results = {}
for model_name, model in models.items():
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Calculate metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    mae_train = mean_absolute_error(y_train, pred_train)
    r2_train = r2_score(y_train, pred_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    mae_test = mean_absolute_error(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)

    results[model_name] = {
        'RMSE (Train)': rmse_train,
        'MAE (Train)': mae_train,
        'R² (Train)': r2_train,
        'RMSE (Test)': rmse_test,
        'MAE (Test)': mae_test,
        'R² (Test)': r2_test
    }

# Print performance metrics for all models
for model_name, metrics in results.items():
    print(f'\nPerformance metrics for {model_name}:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')

# Plotting Actual vs Predicted values for all models
plt.figure(figsize=(14, 10))
for i, (model_name, model) in enumerate(models.items(), 1):
    plt.subplot(2, 3, i)
    pred_test = model.predict(X_test)
    plt.plot(y_test.index, y_test, label='Actual Close Price (Test)', color='blue')
    plt.plot(y_test.index, pred_test, label=f'Predicted ({model_name})', color='red')
    plt.title(f'Actual vs Predicted Close Prices ({model_name})')
    plt.xlabel('Datetime')
    plt.ylabel('Price')
    plt.legend()

plt.tight_layout()
plt.savefig('picture/actual_vs_predicted_close_prices_all_models.png')
plt.close()

# Save the best models using torch
torch.save(best_model_rf, 'model/best_rf_model_close.pt')
torch.save(best_model_xgb, 'model/best_xgb_model_close.pt')
torch.save(best_model_catboost, 'model/best_catboost_model_close.pt')
torch.save(mix_model, 'model/best_mix_model_close.pt')
