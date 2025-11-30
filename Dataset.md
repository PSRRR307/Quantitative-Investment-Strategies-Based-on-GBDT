# **Hang Seng Tech Index Dataset Documentation**

## **1\. Overview**

This dataset contains historical market data for the **Hang Seng Tech Index**, used for the "Hang Seng Tech Index Prediction and Quantitative Strategy Project." The data is structured to support machine learning models (GBDT, Random Forest, XGBoost, etc.) and quantitative backtesting.  
It includes two primary time resolutions:

1. **Daily Data:** End-of-day summary statistics.  
2. **5-Minute Data:** High-frequency intraday data.

## **2\. Data Source & Location**

* **Online Root Source:** Investing.com
* **Input Directory:** optimal/online\_data/ and HangSengTechIndex\_data/.  
* **Processed Directory:** preprocessed\_data/ and optimal/preprocessed\_data/.

## **3\. License**

* **Usage:** This dataset is provided for **Research and Educational Purposes Only**.  
* **Copyright:** Underlying market data rights belong to the original exchange or data provider. Commercial redistribution is restricted.

## **4\. Dataset Features**

### **4.1. Raw Features (Input)**

The raw data files (daily.xlsx and 5min.xlsx) contain the following columns:

| Feature Name | Type | Description |
| :---- | :---- | :---- |
| **Datetime** | Timestamp | The date (and time for 5-min data) of the record. (Format: YYYY-MM-DD or YYYY-MM-DD HH:MM) |
| **Time** | Integer | (5-min only) The specific time interval identifier (e.g., 1355 for 13:55). |
| **Open** | Float | The opening price of the index for the given period. |
| **High** | Float | The highest price reached during the period. |
| **Low** | Float | The lowest price reached during the period. |
| **Close** | Float | The closing price of the index for the given period. |
| **Volume** | Integer | The number of shares/units traded during the period. |

### **4.2. Engineered Features (Processed)**

During the Data Preprocessing and Factor Mining stages, the following technical indicators are generated to serve as input features for the ML models:

* **Moving Averages:** Trend indicators calculated over sliding windows.  
* **RSI (Relative Strength Index):** Momentum oscillator measuring speed and change of price movements.  
* **Bollinger Bands:** Volatility indicators.  Represents the price's position within the Bollinger Bands. Formula: Percent = (Price - Lower Band) / (Upper Band - Lower Band). Higher percentage indicates price is closer to the upper band (potential overbought); conversely for oversold.
* **Normalization:** Data is scaled to ensure model stability.
* **Volume:** Represents the quantity of assets (e.g., number of shares) traded within a specific time period (e.g., one day). Volume is an important indicator of market activity, usually used in conjunction with price trends; high volume may indicate trend strength or turning points.
* **Return:** Represents the profit of an asset over a certain period, which can be absolute return (price change) or relative return (percentage change). The formula is: (Current Price - Previous Price) / Previous Price. Used to assess asset profitability.
* **Amplitude:** Usually refers to the difference between the highest and lowest prices within a certain period (e.g., one day). Larger amplitude indicates more intense price fluctuations, possibly reflecting unstable market sentiment.
* **Volume Change:** Represents the difference between current volume and the volume of the previous period, usually expressed as a percentage. Used to observe volume increases/decreases and judge changes in market sentiment (e.g., increased volume may indicate strengthening trend).
* **20-day Standard Deviation:** Used to measure the magnitude of price fluctuations; the 20-day standard deviation represents the average level of price fluctuations over the past 20 days. Standard deviation measures risk; the larger it is, the greater the price fluctuation and risk.
* **5-Day Momentum:** Difference between current price and price 5 days ago
* **10-day Momentum:** Difference between current price and price 10 days ago
* **20-day Volatility:** Usually represented by the standard deviation of returns over 20 days, measuring the degree of price fluctuation

## **5\. Data Statistics & Range**

Based on the provided raw files:

* **Start Date:** January 3, 2022  
* **End Date:** August 2024 (based on daily.xlsx entries)  
* **Granularity:**  
  * Daily records (\~600+ trading days).  
  * 5-minute intervals (High density for intraday strategy).

## **6\. Data Splits**

The project implements a chronological split to prevent data leakage (look-ahead bias) in time-series forecasting.

* **Training Set:** Located in training\_set/ and optimal/training\_set/. Used to train the ensemble models.  
* **Test Set:** Located in test\_set/ and optimal/test\_set/. Used for:  
  1. Model evaluation (Prediction accuracy).  
  2. Backtesting via prime\_strategy.py (calculating ROI and Max Drawdown).

## **7\. Data Cleaning & Preprocessing Steps**

The main.py and preprocessing modules perform the following operations:

1. **Feature Engineering:**  
   * Calculation of technical factors (RSI, MA, etc.) from raw OHLCV data.   
2. **Formatting:**  
   * Conversion of Datetime strings to Python datetime objects for proper time-series indexing.