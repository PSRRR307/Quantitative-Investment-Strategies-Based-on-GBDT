# Hang Seng Tech Index Prediction and Quantitative Strategy Project

##Notice
Notice that the optimal folder is actually the data folder

## Project Overview

A machine learning-based stock price prediction system for the Hang Seng Tech Index, utilizing various ML algorithms for price forecasting and developing quantitative investment strategies.

## Quick Start

1. Data Preprocessing
python pre_process.py

2. View Baseline Results
Baseline model predictions are located at: optimal/prediction_dt/

3. Train GBDT Model
python GBDT.py

4. Run Mixed Model
python mix_model.py

5. Execute Trading Strategy
python prime_strategy.py

6. Run Complete Pipeline
python baseline.py
python main.py

## Single Command Reproducibility
run.sh

#!/bin/bash
echo "Starting Hang Seng Tech Index Prediction Pipeline..."
echo "==================================================="
echo "Step 1: Data preprocessing..."
python pre_process.py
echo "Step 2: Data splitting..."
python split.py
echo "Step 3: Training baseline model..."
python baseline.py
echo "Step 4: Training GBDT model..."
python GBDT.py
echo "Step 5: Training ensemble model..."
python mix_model.py
echo "Step 6: Executing trading strategies..."
python prime_strategy.py
python best_stategy.py
python opt_strategy.py
echo "==================================================="
echo "Pipeline completed successfully!"
echo "Check the following directories for results:"
echo "- Baseline predictions: optimal/prediction_dt/"
echo "- Model predictions: prediction/"
echo "- Visualizations: picture/"
echo "- Strategy results: output_files/"

## Installation

### Requirements
- Python 3.7+
- Dependencies listed in `requirement.txt`

### Installation Steps
pip install -r requirement.txt

## Project Structure
project-root/
├── model/                         # Trained model files
├── optimal/                        
│   ├── online_data/               # data
│   ├── prediction/                # Prediction results
│   ├── prediction_dt/             # Baseline decision tree predictions
│   ├── preprocessed_data/         # Preprocessed data
│   ├── test_set/                  # Test dataset
│   └── training_set/              # Training dataset
├── origin/                        
├── output_files/                  # Output files
├── picture/                       # Generated charts and graphs
├── prediction/                    # prediction results
├── preprocessed_data/             # preprocessed data
├── test_set/                      # test dataset
├── training_set/                  # training dataset
└── __pycache__/                   # Python cache files

## Core Code Files

### Data Processing
pre_process.py - Data preprocessing and feature engineering
read.py - Data reading module
split.py - Data splitting
data_combination.py - Data combination

### Machine Learning Models
baseline.py - Baseline model implementation
GBDT.py - GBDT model training and prediction
mix_model.py - Mixed model ensemble
first_version.py - Initial version code

### Trading Strategies
prime_strategy.py - Main trading strategy
best_stategy.py - Optimal strategy selection
opt_strategy.py - Strategy optimization

###Main Programs
main.py - Project main entry point

## Main Features

Data Preprocessing: Missing value handling, feature engineering, data normalization
Factor Mining: Technical indicator calculation (moving averages, RSI, Bollinger Bands, etc.)
Model Training: GBDT, Random Forest, XGBoost, CatBoost, etc.
Model Ensemble: Linear regression combining multiple base models
Strategy Backtesting: Basic investment strategy and index enhancement strategy
Result Visualization: Prediction results and strategy performance charts


## Notes
Our model's baseline is independent from main.py, so please run both of them
Ensure data files are correctly placed in the origin/ directory
Preprocessed data will be generated in preprocessing directories on first run
Baseline results are saved in optimal/prediction_dt/
Final prediction results are in prediction/ directory
All charts are output to picture/ directory

## Data Access

###I nput Data Location
origin/

### Data Flow Description
Raw Data: origin/ directory
Preprocessed Data: preprocessed_data/ and optimal/preprocessed_data/
Training/Test Sets: training_set/, test_set/ and corresponding optimal subdirectories

### Output Locations
Baseline Predictions: optimal/prediction_dt/
Model Predictions: prediction/ and optimal/prediction/
Training Results: training_set/ directories
Test Results: test_set/ directories
Visualizations: picture/ directory
Strategy Outputs: output_files/ directory
