import numpy as np
import pandas as pd

np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')
n = len(dates)

S0 = 75.0
mu = 0.0003
sigma = 0.018
returns = np.random.normal(mu, sigma, n)

crash_idx = np.where((dates >= '2020-02-15') & (dates <= '2020-03-23'))[0]
returns[crash_idx] -= 0.025
recovery_idx = np.where((dates >= '2020-03-24') & (dates <= '2020-09-01'))[0]
returns[recovery_idx] += 0.006
bear_idx = np.where((dates >= '2022-01-01') & (dates <= '2022-10-15'))[0]
returns[bear_idx] -= 0.003

prices = S0 * np.exp(np.cumsum(returns))
high   = prices * (1 + np.abs(np.random.normal(0, 0.008, n)))
low    = prices * (1 - np.abs(np.random.normal(0, 0.008, n)))
open_  = prices * (1 + np.random.uniform(-0.005, 0.005, n))
volume = np.random.randint(50_000_000, 200_000_000, n)

df = pd.DataFrame({
    'Date': dates,
    'Open': np.round(open_, 2),
    'High': np.round(high, 2),
    'Low':  np.round(low, 2),
    'Close': np.round(prices, 2),
    'Volume': volume
})
df.to_csv('data/AAPL_stock_data.csv', index=False)
print(f"Generated {len(df)} rows")
print(df.tail())
