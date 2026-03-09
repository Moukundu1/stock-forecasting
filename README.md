# 📈 Stock Price Trend Forecasting — AAPL (2020–2024)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Domain](https://img.shields.io/badge/Domain-Finance%20%2F%20Stock%20Market-purple)

> A comprehensive **end-to-end Machine Learning project** for forecasting next-day stock closing prices using technical indicators, time series feature engineering, and multiple regression models — with rigorous walk-forward validation to prevent data leakage.

---

## 📌 Project Overview

Stock price prediction is one of the most challenging and widely studied problems in quantitative finance. This project builds a complete forecasting pipeline on **5 years of AAPL stock data (2020–2024)**, covering:

- Realistic OHLCV data simulation using **Geometric Brownian Motion** (with COVID crash and 2022 bear market patterns)
- Rich **technical indicator feature engineering** (35+ features)
- Training and evaluating **5 regression models** side-by-side
- **Walk-Forward Validation** — the correct way to validate time series models
- **Golden Cross / Death Cross** trading signal detection
- Complete residual and model diagnostics

---

## 🎯 Objectives

- Engineer meaningful technical features from raw OHLCV data
- Train and compare Linear, Ridge, Lasso, Random Forest, and Gradient Boosting regressors
- Validate models using time-aware walk-forward cross-validation (no data leakage)
- Identify the best-performing model for next-day price prediction
- Generate interpretable visualizations for analysis and reporting

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Ticker** | AAPL (Apple Inc.) — synthetic but realistic |
| **Period** | January 2020 – December 2024 |
| **Frequency** | Business days only (~1,305 rows) |
| **Columns** | Date, Open, High, Low, Close, Volume |
| **Simulation** | Geometric Brownian Motion with real market events (COVID crash, 2022 bear market, 2023–24 recovery) |

> 💡 The data is synthetically generated to mirror real AAPL price behavior. You can replace `data/AAPL_stock_data.csv` with real data downloaded via `yfinance`.

---

## 🔧 Feature Engineering (35+ Features)

### Price-Based Features
| Feature | Description |
|---------|-------------|
| `SMA_5/10/20/50/200` | Simple Moving Averages |
| `EMA_5/10/20` | Exponential Moving Averages |
| `BB_Upper/Lower/Width/Pct` | Bollinger Bands (20-day) |
| `Price_vs_SMA20/50` | Price deviation from moving averages |

### Momentum / Oscillator Features
| Feature | Description |
|---------|-------------|
| `RSI_14` | Relative Strength Index — overbought/oversold |
| `MACD` | Moving Average Convergence Divergence |
| `MACD_Signal` | 9-day EMA of MACD |
| `MACD_Hist` | MACD Histogram (momentum direction) |

### Volatility Features
| Feature | Description |
|---------|-------------|
| `Volatility_10/30` | Rolling standard deviation of daily returns |
| `ATR_14` | Average True Range — market volatility |
| `High_Low_Pct` | Intraday range as percentage of close |

### Volume Features
| Feature | Description |
|---------|-------------|
| `Volume_SMA_20` | 20-day average volume |
| `Volume_Ratio` | Today's volume vs 20-day average |
| `OBV` | On-Balance Volume — buying/selling pressure |

### Lag Features
| Feature | Description |
|---------|-------------|
| `Close_Lag_1/2/3/5` | Closing prices from previous N days |
| `Return_Lag_1/2/3` | Daily returns from previous N days |

### Time Features
`DayOfWeek`, `Month`, `Quarter`, `Year`

---

## 🤖 Models

| Model | Type | Key Strength |
|-------|------|-------------|
| **Linear Regression** | Linear | Interpretable, fast, strong baseline |
| **Ridge Regression** | Regularized Linear | Handles multicollinearity with L2 penalty |
| **Lasso Regression** | Regularized Linear | Automatic feature selection with L1 penalty |
| **Random Forest** | Ensemble (Bagging) | Captures non-linear patterns, robust to outliers |
| **Gradient Boosting** | Ensemble (Boosting) | High accuracy with sequential error correction |

---

## 📉 Model Results

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² |
|-------|-----------|------------|----------|----|
| **Linear Regression** | **2.06** | **2.62** | **1.49%** | **0.9623** |
| Lasso Regression | 2.10 | 2.64 | 1.51% | 0.9615 |
| Ridge Regression | 2.30 | 2.87 | 1.65% | 0.9544 |
| Random Forest | 26.82 | 29.67 | 18.52% | -3.86 |
| Gradient Boosting | 28.43 | 31.31 | 19.65% | -4.41 |

> 🔍 **Key Insight:** Linear models outperform tree-based models here because stock prices are highly autocorrelated — tomorrow's price is very close to today's price. Tree-based models struggle when the test set price range extends beyond what was seen in training (extrapolation problem).

### Walk-Forward Validation (5 Folds)

| Fold | Train Size | MAE | RMSE | R² |
|------|-----------|-----|------|----|
| 1 | 552 | 1.49 | 1.71 | 0.820 |
| 2 | 662 | 1.16 | 1.45 | 0.982 |
| 3 | 772 | 1.46 | 1.79 | 0.971 |
| 4 | 882 | 2.09 | 2.61 | 0.968 |
| 5 | 992 | 2.11 | 2.65 | 0.879 |
| **Avg** | — | **1.66** | **2.04** | **0.924** |

---

## 📁 Project Structure

```
stock-forecasting/
│
├── data/
│   ├── generate_data.py           # Synthetic data generator (GBM simulation)
│   └── AAPL_stock_data.csv        # Raw OHLCV data (2020–2024)
│
├── src/
│   ├── 01_preprocessing.py        # Feature engineering + EDA + charts 1–3
│   ├── 02_modeling.py             # Model training, evaluation + charts 4–7
│   └── 03_timeseries_analysis.py  # Walk-forward validation + charts 8–10
│
├── notebooks/
│   └── stock_forecasting.ipynb    # Full interactive Jupyter Notebook
│
├── outputs/
│   ├── featured_data.csv          # Engineered feature dataset
│   ├── model_comparison.csv       # Metrics for all 5 models
│   ├── 01_price_overview.png      # Price + SMA + Bollinger + RSI + MACD
│   ├── 02_returns_volatility.png  # Returns distribution & cumulative return
│   ├── 03_correlation_heatmap.png # Feature correlation matrix
│   ├── 04_actual_vs_predicted.png # Best model predictions vs actual
│   ├── 05_all_models_comparison.png # All 5 models side-by-side
│   ├── 06_feature_importance.png  # Random Forest feature importances
│   ├── 07_metrics_dashboard.png   # RMSE / MAE / R² bar charts
│   ├── 08_walk_forward_validation.png # Walk-forward results
│   ├── 09_trading_signals.png     # Golden Cross / Death Cross chart
│   └── 10_residual_analysis.png   # Residuals, histogram, Q-Q plot
│
├── models/
│   └── *.pkl                      # Saved trained models (joblib)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Moukundu1/stock-forecasting.git
cd stock-forecasting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data
```bash
python data/generate_data.py
```

> 💡 **To use real data:** Replace this step with:
> ```python
> import yfinance as yf
> df = yf.download('AAPL', start='2020-01-01', end='2024-12-31')
> df.to_csv('data/AAPL_stock_data.csv')
> ```

### 4. Run the Full Pipeline
```bash
# Step 1: Preprocessing & Feature Engineering
python src/01_preprocessing.py

# Step 2: Model Training & Evaluation
python src/02_modeling.py

# Step 3: Time Series Analysis & Walk-Forward Validation
python src/03_timeseries_analysis.py
```

### 5. Or Open the Jupyter Notebook
```bash
jupyter notebook notebooks/stock_forecasting.ipynb
```

---

## 📸 Output Visualizations

The pipeline generates **10 charts** saved in `outputs/`:

1. **Price Overview** — Closing price, SMAs, Bollinger Bands, Volume, RSI, MACD
2. **Returns & Volatility** — Daily returns, distribution, cumulative return, rolling volatility
3. **Correlation Heatmap** — Feature correlation matrix (lower triangle)
4. **Actual vs Predicted** — Best model predictions on the test set
5. **All Models Comparison** — 5-panel comparison of all models
6. **Feature Importance** — Top 20 features from Random Forest
7. **Metrics Dashboard** — RMSE, MAE, R² bar charts across all models
8. **Walk-Forward Validation** — Fold-by-fold accuracy analysis
9. **Trading Signals** — Golden Cross & Death Cross on price chart
10. **Residual Analysis** — Scatter, histogram, Q-Q plot for diagnostics

---

## 💡 Key Concepts & Learnings

- **Why Linear beats Tree models for price prediction:** Tree-based models cannot extrapolate beyond training data ranges — they predict within the min/max of what they've seen. Since stock prices trend upward over time, tree models fail on future data.
- **Walk-Forward Validation:** The only correct way to validate time series models. Standard k-fold uses future data to predict the past — which is data leakage.
- **Technical Indicators:** RSI, MACD, Bollinger Bands are industry-standard signals used by quants and traders worldwide.
- **Feature Engineering matters more than model choice:** With 35 well-engineered features, even a simple Linear Regression achieves R² > 0.96.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations, GBM simulation |
| **Scikit-learn** | ML models, preprocessing, metrics |
| **Matplotlib / Seaborn** | Visualization |
| **SciPy** | Statistical analysis (Q-Q plot) |
| **Joblib** | Model serialization |
| **yfinance** | Real stock data (optional) |
| **Jupyter** | Interactive exploration |

---

## 🔮 Future Improvements

- [ ] Add **LSTM / GRU** (deep learning) for sequence modeling
- [ ] Add **Prophet** for seasonality decomposition
- [ ] Implement **Hyperparameter tuning** (GridSearchCV with TimeSeriesSplit)
- [ ] Add **direction prediction** (classification: will price go up or down?)
- [ ] Build a **Streamlit dashboard** for interactive predictions
- [ ] Integrate **real-time data** via yfinance or Alpha Vantage API

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/lstm-model`)
3. Commit your changes (`git commit -m 'Add LSTM model'`)
4. Push to the branch (`git push origin feature/lstm-model`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙋 Author

Built as part of a Data Science portfolio.  
Connect on [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

---

⭐ **If this project was useful, please star the repository!** ⭐
