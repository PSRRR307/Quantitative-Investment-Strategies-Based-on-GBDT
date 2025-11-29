import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Load and preprocess data
def load_data(training_path, test_path):
    training_data = pd.read_excel(training_path)
    test_data = pd.read_excel(test_path)
    training_data['Datetime'] = pd.to_datetime(training_data['Datetime'])
    training_data.set_index('Datetime', inplace=True)
    test_data['Datetime'] = pd.to_datetime(test_data['Datetime'])
    test_data.set_index('Datetime', inplace=True)
    return training_data, test_data


# Create lagged features
def create_lagged_features(data, lag=1):
    lagged_data = data.copy()
    for i in range(1, lag + 1):
        lagged_data[f'Open_Lag_{i}'] = lagged_data['Open'].shift(i)
        lagged_data[f'High_Lag_{i}'] = lagged_data['High'].shift(i)
        lagged_data[f'Low_Lag_{i}'] = lagged_data['Low'].shift(i)
        lagged_data[f'Close_Lag_{i}'] = lagged_data['Close'].shift(i)
    return lagged_data


# Prepare features and target variables
def prepare_data(lagged_data):
    X = lagged_data.drop(columns=['Open', 'High', 'Low', 'Close'])
    y = lagged_data[['Open', 'High', 'Low', 'Close']]
    return X, y


# Hyperparameter tuning using GridSearchCV
def tune_hyperparameters(X_train, y_train, param_grid, tscv):
    gbdt_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=gbdt_model,
                               param_grid=param_grid,
                               scoring='neg_root_mean_squared_error',
                               cv=tscv,
                               verbose=1,
                               n_jobs=-1,
                               error_score=np.nan)
    best_models = {}
    best_rmse_values = {}
    for target in ['Open', 'High', 'Low', 'Close']:
        grid_search.fit(X_train, y_train[target])
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        best_model = GradientBoostingRegressor(**best_params)
        best_model.fit(X_train, y_train[target])
        best_models[target] = best_model
        best_rmse_values[target] = best_score
    return best_models, best_rmse_values


# Plot learning curves
def plot_learning_curves(models, X, y, tscv):
    plt.figure(figsize=(15, 12))
    for i, target in enumerate(['Open', 'High', 'Low', 'Close']):
        train_sizes, train_scores, val_scores = learning_curve(
            models[target],
            X,
            y[target],
            train_sizes=np.linspace(0.1, 1.0, 40),
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        train_scores_mean = -train_scores.mean(axis=1)
        val_scores_mean = -val_scores.mean(axis=1)
        plt.subplot(4, 1, i + 1)
        plt.plot(train_sizes, train_scores_mean, label='Training RMSE', color='blue')
        plt.plot(train_sizes, val_scores_mean, label='Validation RMSE', color='red')
        plt.title(f'Learning Curve for {target}')
        plt.xlabel('Training Size')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()


# Evaluate model performance
def evaluate_model(best_models, X_train, y_train, X_test, y_test):
    predictions_train = np.zeros((X_train.shape[0], 4))
    predictions_test = np.zeros((X_test.shape[0], 4))
    for i, target in enumerate(['Open', 'High', 'Low', 'Close']):
        predictions_train[:, i] = best_models[target].predict(X_train)
        predictions_test[:, i] = best_models[target].predict(X_test)
    for i, target in enumerate(['Open', 'High', 'Low', 'Close']):
        actuals_train = y_train[target].values
        actuals_test = y_test[target].values
        rmse_train = np.sqrt(mean_squared_error(actuals_train, predictions_train[:, i]))
        mae_train = mean_absolute_error(actuals_train, predictions_train[:, i])
        r2_train = r2_score(actuals_train, predictions_train[:, i])
        rmse_test = np.sqrt(mean_squared_error(actuals_test, predictions_test[:, i]))
        mae_test = mean_absolute_error(actuals_test, predictions_test[:, i])
        r2_test = r2_score(actuals_test, predictions_test[:, i])
        print(f'Performance metrics for {target}:')
        print(f'Training RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.4f}')
        print(f'Test RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.4f}\n')
    return predictions_test


# Plot actual vs predicted values (仅针对 Close)
def plot_predictions(y_test, predictions, stock_code):
    plt.figure(figsize=(10, 6))
    target = 'Close'  # 仅针对 Close
    plt.plot(y_test.index, y_test[target], label='Actual Price (Test)', color='blue')
    plt.plot(y_test.index, predictions[:, 3], label='Predicted Price (Test)', color='red')  # Close 是第4列
    plt.title(f'Actual vs Predicted {target} Prices using GBDT - {stock_code}')
    plt.xlabel('Datetime')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'picture/predictions_{stock_code}.png')
    plt.close()


def save(y_test, predictions,path):
    # 创建包含预测结果的DataFrame
    predicted_data = pd.DataFrame({
        'Datetime': y_test.index,
        'Open': predictions[:, 0],
        'High': predictions[:, 1],
        'Low': predictions[:, 2],
        'Close': predictions[:, 3]
    })

    # 保存为.xlsx文件
    output_file = path
    with pd.ExcelWriter(output_file) as writer:
        predicted_data.to_excel(writer, sheet_name='Sheet1', index=False)

    print(f'预测结果已保存到 {output_file}')


# Main function to orchestrate the workflow
def gbdt_main(training_path, test_path, save_path,stock_name):
    # Load data
    training_data, test_data = load_data(training_path, test_path)

    # Create lagged features
    lag = 1
    training_lagged = create_lagged_features(training_data, lag)
    test_lagged = create_lagged_features(test_data, lag)

    # Drop NaN values
    training_lagged.dropna(inplace=True)
    test_lagged.dropna(inplace=True)

    # Prepare features and target variables
    X_train, y_train = prepare_data(training_lagged)
    X_test, y_test = prepare_data(test_lagged)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50],
        'learning_rate': [0.1],
        'max_depth': [5],
        'min_samples_split': [5],
        'min_samples_leaf': [2]
    }

    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5, gap=20, test_size=20)

    # Tune hyperparameters
    best_models, best_rmse_values = tune_hyperparameters(X_train, y_train, param_grid, tscv)

    # # Plot learning curves
    # plot_learning_curves(best_models, X_train, y_train, tscv)

    # Evaluate model performance
    predictions_test = evaluate_model(best_models, X_train, y_train, X_test, y_test)

    # Plot actual vs predicted values
    plot_predictions(y_test, predictions_test,stock_name)

    save(y_test, predictions_test, save_path)


# Run the main function
if __name__ == "__main__":
    training_path = "training_set\daily.xlsx"
    test_path = "test_set\daily.xlsx"
    save_path = "prediction\predictions.xlsx"
    gbdt_main(training_path, test_path, save_path)
