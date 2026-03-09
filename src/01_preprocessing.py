"""
============================================================
  Stock Price Trend Forecasting
  Module: Preprocessing & Feature Engineering
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='darkgrid')
os.makedirs('outputs', exist_ok=True)

# ────────────────────────────────────────────
# 1. LOAD & VALIDATE DATA
# ────────────────────────────────────────────
print("=" * 65)
print("  STOCK PRICE FORECASTING — AAPL 2020–2024")
print("=" * 65)

df = pd.read_csv('data/AAPL_stock_data.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n✅ Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"   Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"\n📋 Missing values:\n{df.isnull().sum()}")

# ────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ────────────────────────────────────────────
print("\n" + "─" * 65)
print("FEATURE ENGINEERING")
print("─" * 65)

# --- Returns ---
df['Daily_Return']    = df['Close'].pct_change()
df['Log_Return']      = np.log(df['Close'] / df['Close'].shift(1))
df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1

# --- Moving Averages ---
for w in [5, 10, 20, 50, 200]:
    df[f'SMA_{w}']  = df['Close'].rolling(w).mean()
    df[f'EMA_{w}']  = df['Close'].ewm(span=w, adjust=False).mean()

# --- Bollinger Bands (20-day) ---
df['BB_Mid']   = df['SMA_20']
df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(20).std()
df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(20).std()
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
df['BB_Pct']   = (df['Close'] - df['BB_Lower']) / df['BB_Width']

# --- RSI (14-day) ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI_14'] = compute_rsi(df['Close'])

# --- MACD ---
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

# --- Volatility ---
df['Volatility_10']  = df['Daily_Return'].rolling(10).std()
df['Volatility_30']  = df['Daily_Return'].rolling(30).std()
df['ATR_14'] = (df['High'] - df['Low']).rolling(14).mean()   # simplified ATR

# --- Price Position Features ---
df['High_Low_Pct']   = (df['High'] - df['Low']) / df['Close']
df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

# --- Lag Features ---
for lag in [1, 2, 3, 5, 10]:
    df[f'Close_Lag_{lag}']  = df['Close'].shift(lag)
    df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

# --- Volume Features ---
df['Volume_SMA_20']  = df['Volume'].rolling(20).mean()
df['Volume_Ratio']   = df['Volume'] / df['Volume_SMA_20']
df['OBV'] = (np.sign(df['Daily_Return']) * df['Volume']).cumsum()

# --- Time Features ---
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month']     = df['Date'].dt.month
df['Quarter']   = df['Date'].dt.quarter
df['Year']      = df['Date'].dt.year

# --- Target Variable ---
df['Target_Close']    = df['Close'].shift(-1)           # next day close
df['Target_Direction'] = (df['Target_Close'] > df['Close']).astype(int)  # 1=up, 0=down

print(f"✅ Features created: {df.shape[1]} total columns")

# Drop rows with NaN (from rolling windows + lag)
df_clean = df.dropna().reset_index(drop=True)
print(f"✅ After dropping NaN rows: {df_clean.shape[0]} rows remain")

df_clean.to_csv('outputs/featured_data.csv', index=False)
print("✅ Saved → outputs/featured_data.csv")

# ────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ────────────────────────────────────────────
print("\n" + "─" * 65)
print("EXPLORATORY DATA ANALYSIS")
print("─" * 65)

# ── Fig 1: Price + Volume + RSI Overview ──
fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('AAPL Stock — Full Overview (2020–2024)', fontsize=15, fontweight='bold', y=0.98)

axes[0].plot(df_clean['Date'], df_clean['Close'], color='#1f77b4', linewidth=1.2, label='Close')
axes[0].plot(df_clean['Date'], df_clean['SMA_50'], color='orange', linewidth=1, linestyle='--', alpha=0.8, label='SMA 50')
axes[0].plot(df_clean['Date'], df_clean['SMA_200'], color='red', linewidth=1, linestyle='--', alpha=0.8, label='SMA 200')
axes[0].fill_between(df_clean['Date'], df_clean['BB_Lower'], df_clean['BB_Upper'], alpha=0.1, color='blue', label='Bollinger Bands')
axes[0].set_ylabel('Price (USD)')
axes[0].legend(loc='upper left', fontsize=8)
axes[0].set_title('Closing Price with Moving Averages & Bollinger Bands')

axes[1].bar(df_clean['Date'], df_clean['Volume'] / 1e6, color='steelblue', alpha=0.6, width=1)
axes[1].plot(df_clean['Date'], df_clean['Volume_SMA_20'] / 1e6, color='red', linewidth=1, label='20-day Avg Volume')
axes[1].set_ylabel('Volume (M)')
axes[1].set_title('Trading Volume')
axes[1].legend(fontsize=8)

rsi = df_clean['RSI_14']
axes[2].plot(df_clean['Date'], rsi, color='purple', linewidth=1)
axes[2].axhline(70, color='red', linestyle='--', alpha=0.7, linewidth=0.8, label='Overbought (70)')
axes[2].axhline(30, color='green', linestyle='--', alpha=0.7, linewidth=0.8, label='Oversold (30)')
axes[2].fill_between(df_clean['Date'], rsi, 70, where=(rsi >= 70), alpha=0.3, color='red')
axes[2].fill_between(df_clean['Date'], rsi, 30, where=(rsi <= 30), alpha=0.3, color='green')
axes[2].set_ylabel('RSI')
axes[2].set_title('RSI (14-day)')
axes[2].legend(fontsize=8)

axes[3].plot(df_clean['Date'], df_clean['MACD'], color='blue', linewidth=1, label='MACD')
axes[3].plot(df_clean['Date'], df_clean['MACD_Signal'], color='red', linewidth=1, label='Signal')
axes[3].bar(df_clean['Date'], df_clean['MACD_Hist'],
            color=['green' if x > 0 else 'red' for x in df_clean['MACD_Hist']],
            alpha=0.5, width=1, label='Histogram')
axes[3].set_ylabel('MACD')
axes[3].set_title('MACD Indicator')
axes[3].legend(fontsize=8)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('outputs/01_price_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/01_price_overview.png")

# ── Fig 2: Returns & Volatility ──
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Returns & Volatility Analysis', fontsize=14, fontweight='bold')

axes[0,0].plot(df_clean['Date'], df_clean['Daily_Return'] * 100, color='steelblue', linewidth=0.8, alpha=0.8)
axes[0,0].axhline(0, color='black', linewidth=0.8)
axes[0,0].set_title('Daily Returns (%)')
axes[0,0].set_ylabel('Return (%)')

axes[0,1].hist(df_clean['Daily_Return'].dropna() * 100, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
axes[0,1].axvline(0, color='red', linestyle='--')
axes[0,1].set_title('Return Distribution')
axes[0,1].set_xlabel('Daily Return (%)')

axes[1,0].plot(df_clean['Date'], df_clean['Volatility_30'] * 100, color='crimson', linewidth=1)
axes[1,0].set_title('30-Day Rolling Volatility (%)')
axes[1,0].set_ylabel('Volatility (%)')

axes[1,1].plot(df_clean['Date'], df_clean['Cumulative_Return'] * 100, color='darkgreen', linewidth=1.5)
axes[1,1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1,1].set_title('Cumulative Return (%)')
axes[1,1].set_ylabel('Cumulative Return (%)')

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('outputs/02_returns_volatility.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/02_returns_volatility.png")

# ── Fig 3: Feature Correlation Heatmap ──
corr_cols = ['Close', 'Daily_Return', 'RSI_14', 'MACD', 'BB_Pct',
             'Volume_Ratio', 'Volatility_30', 'Price_vs_SMA20',
             'Price_vs_SMA50', 'High_Low_Pct', 'ATR_14', 'OBV']
corr_matrix = df_clean[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved → outputs/03_correlation_heatmap.png")

print(f"\n{'='*65}")
print(f"  PREPROCESSING COMPLETE — {df_clean.shape[1]} features ready for modeling")
print(f"{'='*65}")
