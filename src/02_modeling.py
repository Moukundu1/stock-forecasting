"""
============================================================
  Stock Price Trend Forecasting
  Module: ML Modeling — Linear vs Tree-Based Models
  Models: Linear Regression, Ridge, Random Forest, Gradient Boosting
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model   import LinearRegression, Ridge, Lasso
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing  import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics        import (mean_absolute_error, mean_squared_error,
                                    r2_score, mean_absolute_percentage_error)
import joblib

sns.set_theme(style='darkgrid')
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ────────────────────────────────────────────
# 1. LOAD FEATURED DATA
# ────────────────────────────────────────────
print("=" * 65)
print("  ML MODELING — STOCK PRICE REGRESSION")
print("=" * 65)

df = pd.read_csv('outputs/featured_data.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"\n✅ Loaded featured dataset: {df.shape}")

# ────────────────────────────────────────────
# 2. FEATURE SELECTION
# ────────────────────────────────────────────
feature_cols = [
    'Open', 'High', 'Low', 'Volume',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_5', 'EMA_10', 'EMA_20',
    'BB_Width', 'BB_Pct',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'Volatility_10', 'Volatility_30', 'ATR_14',
    'High_Low_Pct', 'Price_vs_SMA20', 'Price_vs_SMA50',
    'Volume_Ratio', 'OBV',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
    'DayOfWeek', 'Month', 'Quarter'
]

TARGET = 'Target_Close'

X = df[feature_cols].copy()
y = df[TARGET].copy()

print(f"✅ Features: {len(feature_cols)} | Target: {TARGET}")

# ────────────────────────────────────────────
# 3. TIME-SERIES SPLIT (No data leakage)
# ────────────────────────────────────────────
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df['Date'].iloc[split_idx:].reset_index(drop=True)

print(f"\n📅 Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"   Train period: {df['Date'].iloc[0].date()} → {df['Date'].iloc[split_idx-1].date()}")
print(f"   Test  period: {df['Date'].iloc[split_idx].date()} → {df['Date'].iloc[-1].date()}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler, 'models/feature_scaler.pkl')

# ────────────────────────────────────────────
# 4. TRAIN MODELS
# ────────────────────────────────────────────
print("\n" + "─" * 65)
print("TRAINING MODELS")
print("─" * 65)

models = {
    'Linear Regression':      LinearRegression(),
    'Ridge Regression':       Ridge(alpha=10.0),
    'Lasso Regression':       Lasso(alpha=0.1, max_iter=5000),
    'Random Forest':          RandomForestRegressor(n_estimators=200, max_depth=10,
                                                     min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Gradient Boosting':      GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                          max_depth=4, subsample=0.8, random_state=42),
}

results   = {}
preds_all = {}

for name, model in models.items():
    # Linear models → scaled features; tree models → raw features
    if 'Regression' in name:
        model.fit(X_train_sc, y_train)
        pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = mean_absolute_percentage_error(y_test, pred) * 100
    r2   = r2_score(y_test, pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'R²': r2}
    preds_all[name] = pred

    joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")
    print(f"  ✅ {name:<25} MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}")

# ────────────────────────────────────────────
# 5. RESULTS TABLE
# ────────────────────────────────────────────
results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values('RMSE')
results_df.to_csv('outputs/model_comparison.csv')

print(f"\n📊 Model Comparison (sorted by RMSE):")
print(results_df.to_string())

best_model_name = results_df.index[0]
best_pred       = preds_all[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")

# ────────────────────────────────────────────
# 6. VISUALIZATIONS
# ────────────────────────────────────────────

# ── Fig 4: Actual vs Predicted (best model) ──
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle(f'Actual vs Predicted — {best_model_name}', fontsize=14, fontweight='bold')

axes[0].plot(dates_test, y_test.values, color='#1f77b4', linewidth=1.5, label='Actual Close')
axes[0].plot(dates_test, best_pred,     color='#ff7f0e', linewidth=1.2, linestyle='--', label=f'Predicted ({best_model_name})')
axes[0].set_ylabel('Price (USD)')
axes[0].set_title('Next-Day Close Price: Actual vs Predicted')
axes[0].legend()

residuals = y_test.values - best_pred
axes[1].bar(dates_test, residuals, color=['green' if r > 0 else 'red' for r in residuals],
            alpha=0.6, width=1)
axes[1].axhline(0, color='black', linewidth=1)
axes[1].set_ylabel('Residual (USD)')
axes[1].set_title('Prediction Residuals')

for ax in axes:
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('outputs/04_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved → outputs/04_actual_vs_predicted.png")

# ── Fig 5: All models comparison ──
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('All Models — Actual vs Predicted (Test Set)', fontsize=14, fontweight='bold')
axes = axes.flat

for i, (name, pred) in enumerate(preds_all.items()):
    axes[i].plot(dates_test, y_test.values, color='steelblue', linewidth=1.2, label='Actual', alpha=0.9)
    axes[i].plot(dates_test, pred, color='crimson', linewidth=1, linestyle='--', label='Predicted', alpha=0.9)
    r2 = results[name]['R²']
    rmse = results[name]['RMSE']
    axes[i].set_title(f'{name}\nR²={r2:.4f} | RMSE={rmse:.2f}', fontsize=9)
    axes[i].legend(fontsize=7)
    axes[i].tick_params(axis='x', rotation=20, labelsize=7)

# Metrics bar chart in last panel
metrics_to_plot = results_df['RMSE']
axes[5].barh(metrics_to_plot.index, metrics_to_plot.values,
             color=['gold' if i == 0 else 'steelblue' for i in range(len(metrics_to_plot))])
axes[5].set_title('Model RMSE Comparison\n(lower is better)', fontsize=9)
axes[5].set_xlabel('RMSE')

plt.tight_layout()
plt.savefig('outputs/05_all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/05_all_models_comparison.png")

# ── Fig 6: Feature Importance (Random Forest) ──
rf_model = models['Random Forest']
feat_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
top20 = feat_imp.head(20)
colors = ['gold' if i < 3 else 'steelblue' for i in range(len(top20))]
ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
ax.set_title('Top 20 Feature Importances — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/06_feature_importance.png")

# ── Fig 7: Metrics Dashboard ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Performance Metrics', fontsize=14, fontweight='bold')

colors = ['gold', 'steelblue', 'steelblue', 'steelblue', 'steelblue']
model_names_short = [n.replace(' Regression','').replace(' ','  ') for n in results_df.index]

for ax, metric in zip(axes, ['RMSE', 'MAE', 'R²']):
    vals = results_df[metric].values
    bars = ax.bar(model_names_short, vals, color=colors, edgecolor='white')
    ax.set_title(f'{metric} ({"lower better" if metric != "R²" else "higher better"})')
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001 * bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('outputs/07_metrics_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/07_metrics_dashboard.png")

print(f"\n{'='*65}")
print(f"  🏆 BEST MODEL: {best_model_name}")
print(f"     MAE  = {results[best_model_name]['MAE']:.4f}")
print(f"     RMSE = {results[best_model_name]['RMSE']:.4f}")
print(f"     MAPE = {results[best_model_name]['MAPE (%)']:.2f}%")
print(f"     R²   = {results[best_model_name]['R²']:.4f}")
print(f"\n  All models saved in models/")
print(f"  All charts saved in outputs/")
print(f"{'='*65}")
