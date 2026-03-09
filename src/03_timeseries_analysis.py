"""
============================================================
  Stock Price Trend Forecasting
  Module: Time Series Analysis & Walk-Forward Validation
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model  import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style='darkgrid')

# ────────────────────────────────────────────
# 1. LOAD DATA
# ────────────────────────────────────────────
print("=" * 65)
print("  TIME SERIES ANALYSIS & WALK-FORWARD VALIDATION")
print("=" * 65)

df = pd.read_csv('outputs/featured_data.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

feature_cols = [
    'Open', 'High', 'Low', 'Volume',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_5', 'EMA_10', 'EMA_20',
    'BB_Width', 'BB_Pct', 'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Volatility_10', 'Volatility_30', 'ATR_14',
    'High_Low_Pct', 'Price_vs_SMA20', 'Price_vs_SMA50',
    'Volume_Ratio', 'OBV',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
    'DayOfWeek', 'Month', 'Quarter'
]

X = df[feature_cols]
y = df['Target_Close']

# ────────────────────────────────────────────
# 2. WALK-FORWARD VALIDATION
# ────────────────────────────────────────────
print("\n🔄 Running Walk-Forward Validation (5 folds)...")

n = len(X)
min_train = int(n * 0.5)
fold_size = int((n - min_train) / 5)

fold_results = []
all_actuals, all_preds, all_dates = [], [], []

for fold in range(5):
    train_end = min_train + fold * fold_size
    test_start = train_end
    test_end   = min(train_end + fold_size, n)

    X_tr, y_tr = X.iloc[:train_end],        y.iloc[:train_end]
    X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
    d_te = df['Date'].iloc[test_start:test_end]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    model = LinearRegression()
    model.fit(X_tr_sc, y_tr)
    pred = model.predict(X_te_sc)

    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2   = r2_score(y_te, pred)

    fold_results.append({'Fold': fold+1, 'Train Size': train_end,
                         'Test Size': len(y_te), 'MAE': mae, 'RMSE': rmse, 'R²': r2})
    all_actuals.extend(y_te.values)
    all_preds.extend(pred)
    all_dates.extend(d_te.values)

    print(f"  Fold {fold+1} | Train:{train_end:4d} | MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.4f}")

fold_df = pd.DataFrame(fold_results)
print(f"\n📊 Walk-Forward Summary:")
print(fold_df.to_string(index=False))
avg_mae  = fold_df['MAE'].mean()
avg_rmse = fold_df['RMSE'].mean()
avg_r2   = fold_df['R²'].mean()
print(f"\n   Avg MAE={avg_mae:.4f} | Avg RMSE={avg_rmse:.4f} | Avg R²={avg_r2:.4f}")

# ────────────────────────────────────────────
# 3. MOVING WINDOW TREND PREDICTION
# ────────────────────────────────────────────
print("\n📈 Generating trend signals...")

close = df['Close'].values
dates = df['Date'].values

# Golden Cross / Death Cross signals
sma50  = df['SMA_50'].values
sma200 = df['SMA_200'].values

golden_cross = np.where((sma50[1:] > sma200[1:]) & (sma50[:-1] <= sma200[:-1]))[0] + 1
death_cross  = np.where((sma50[1:] < sma200[1:]) & (sma50[:-1] >= sma200[:-1]))[0] + 1

# ────────────────────────────────────────────
# 4. VISUALIZATIONS
# ────────────────────────────────────────────

# ── Fig 8: Walk-Forward Validation ──
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Walk-Forward Validation — Linear Regression', fontsize=14, fontweight='bold')

all_dates_pd = pd.to_datetime(all_dates)
axes[0].plot(all_dates_pd, all_actuals, color='steelblue', linewidth=1.2, label='Actual', alpha=0.9)
axes[0].plot(all_dates_pd, all_preds, color='crimson', linewidth=1, linestyle='--', label='Predicted', alpha=0.8)
axes[0].set_title('Actual vs Predicted — All 5 Folds Combined')
axes[0].set_ylabel('Close Price (USD)')
axes[0].legend()

x = fold_df['Fold'].astype(str)
width = 0.3
axes[1].bar(np.arange(len(x)) - width/2, fold_df['MAE'],  width, label='MAE',  color='steelblue')
axes[1].bar(np.arange(len(x)) + width/2, fold_df['RMSE'], width, label='RMSE', color='coral')
axes[1].set_xticks(np.arange(len(x)))
axes[1].set_xticklabels([f'Fold {i}' for i in x])
axes[1].set_title('MAE & RMSE per Fold')
axes[1].set_ylabel('Error (USD)')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/08_walk_forward_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/08_walk_forward_validation.png")

# ── Fig 9: Golden Cross / Death Cross signals ──
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(dates, close,  color='#1f77b4', linewidth=1.2, label='Close Price', alpha=0.9)
ax.plot(dates, sma50,  color='orange',  linewidth=1.2, linestyle='--', label='SMA 50', alpha=0.85)
ax.plot(dates, sma200, color='red',     linewidth=1.2, linestyle='--', label='SMA 200', alpha=0.85)

if len(golden_cross):
    ax.scatter(dates[golden_cross], close[golden_cross], marker='^', color='lime',
               s=120, zorder=5, label='Golden Cross (Buy)')
if len(death_cross):
    ax.scatter(dates[death_cross], close[death_cross], marker='v', color='red',
               s=120, zorder=5, label='Death Cross (Sell)')

ax.set_title('Golden Cross & Death Cross Signals — AAPL', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('outputs/09_trading_signals.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/09_trading_signals.png")

# ── Fig 10: Residual Analysis ──
split_idx = int(len(X) * 0.80)
X_tr = X.iloc[:split_idx]; y_tr = y.iloc[:split_idx]
X_te = X.iloc[split_idx:]; y_te = y.iloc[split_idx:]
dates_te = df['Date'].iloc[split_idx:]

sc = StandardScaler()
model = LinearRegression()
model.fit(sc.fit_transform(X_tr), y_tr)
pred = model.predict(sc.transform(X_te))
residuals = y_te.values - pred

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Residual Analysis — Linear Regression', fontsize=13, fontweight='bold')

axes[0].scatter(pred, residuals, alpha=0.5, color='steelblue', s=15)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Residual')
axes[0].set_title('Residuals vs Predicted')

axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Residual'); axes[1].set_title('Residual Distribution')

from scipy import stats
stats.probplot(residuals, plot=axes[2])
axes[2].set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig('outputs/10_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/10_residual_analysis.png")

print(f"\n{'='*65}")
print(f"  TIME SERIES ANALYSIS COMPLETE")
print(f"  Walk-Forward Avg RMSE = {avg_rmse:.4f}")
print(f"  Walk-Forward Avg R²   = {avg_r2:.4f}")
print(f"{'='*65}")
