import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def load_and_resample_data(nifty_path):
  print('Loading and resampling Nifty dataset to 5-minute bars...')
  df_nifty = pd.read_csv(nifty_path)
  df_nifty['date'] = pd.to_datetime(df_nifty['date'])
  df_nifty.set_index('date', inplace=True)
  df_nifty.columns = [c.lower() for c in df_nifty.columns]

  df_5min = (
      df_nifty.between_time('09:15', '15:29')
      .resample('5min')
      .agg(
          {
              'open': 'first',
              'high': 'max',
              'low': 'min',
              'close': 'last',
              'volume': 'sum',
          }
      )
      .dropna()
  )
  return df_5min


def engineer_features_5min(df):
  print('Engineering 5-minute features and target labels...')
  df['nifty_ret_1'] = df['close'].pct_change(1)
  df['nifty_ret_3'] = df['close'].pct_change(3)
  df['nifty_ret_6'] = df['close'].pct_change(6)

  df['realized_vol'] = df['nifty_ret_1'].rolling(window=12).std()
  df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
  df['dist_from_ema'] = (df['close'] - df['ema_50']) / df['ema_50']
  df['minutes_from_open'] = (df.index.hour - 9) * 60 + (df.index.minute - 15)

  high_low = df['high'] - df['low']
  high_close = np.abs(df['high'] - df['close'].shift())
  low_close = np.abs(df['low'] - df['close'].shift())
  df['atr'] = (
      pd.concat([high_low, high_close, low_close], axis=1)
      .max(axis=1)
      .rolling(14)
      .mean()
  )

  forward_window = 12
  fwd_high = df['high'].shift(-1).rolling(window=forward_window).max()
  fwd_low = df['low'].shift(-1).rolling(window=forward_window).min()
  df['fwd_range'] = fwd_high - fwd_low

  return df.dropna()


def run_volatility_proxy_backtest(
    df, prob_threshold=0.82, holding_bars=12, option_cost=0.0018
):
  print(
      f'Training model and running Volatility Straddle Proxy Backtest'
      f' (Threshold = {prob_threshold})...'
  )

  feature_cols = [
      'nifty_ret_1',
      'nifty_ret_3',
      'nifty_ret_6',
      'realized_vol',
      'dist_from_ema',
      'minutes_from_open',
  ]

  X = df[feature_cols]
  split_idx = int(len(df) * 0.80)

  train_fwd_range = df['fwd_range'].iloc[:split_idx]
  threshold = train_fwd_range.quantile(0.70)
  y = (df['fwd_range'] >= threshold).astype(int)

  X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
  y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

  model = XGBClassifier(
      n_estimators=100,
      max_depth=5,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      n_jobs=-1,
  )
  model.fit(X_train, y_train)

  test_df = df.iloc[split_idx:].copy()
  test_df['expansion_prob'] = model.predict_proba(X_test)[:, 1]

  print(
      f'Simulating option straddle execution on {len(test_df):,} test'
      ' bars...'
  )
  start_time = time.time()

  test_df['trade_date'] = test_df.index.date
  trade_returns = []
  trade_dates = []
  cutoff_time = pd.to_datetime('14:45:00').time()

  for date, group in test_df.groupby('trade_date'):
    times = group.index.time
    opens = group['open'].values
    highs = group['high'].values
    lows = group['low'].values
    probs = group['expansion_prob'].values

    cooldown = 0

    for i in range(len(group) - holding_bars):
      if times[i] >= cutoff_time:
        continue

      if cooldown > 0:
        cooldown -= 1
        continue

      if probs[i] >= prob_threshold:
        ep = opens[i + 1]
        window_highs = highs[i + 1 : i + 1 + holding_bars]
        window_lows = lows[i + 1 : i + 1 + holding_bars]

        max_excursion = max(
            np.max(window_highs) - ep, ep - np.min(window_lows)
        )
        move_pct = max_excursion / ep

        net_trade_ret = move_pct - option_cost
        trade_returns.append(net_trade_ret)
        trade_dates.append(group.index[i])  # Capture trade timestamp

        cooldown = holding_bars

  trade_returns = np.array(trade_returns)
  total_trades = len(trade_returns)
  winning_trades = np.sum(trade_returns > 0)
  win_rate = winning_trades / total_trades if total_trades > 0 else 0
  avg_trade_ret = np.mean(trade_returns) if total_trades > 0 else 0
  cum_strat_return = np.prod(1 + trade_returns) if total_trades > 0 else 1.0

  print(f'Backtest completed in {time.time() - start_time:.2f} seconds.')
  return total_trades, win_rate, avg_trade_ret, cum_strat_return, trade_dates


if __name__ == '__main__':
  nifty_file = 'nifty_10yr.csv'
  data = load_and_resample_data(nifty_file)
  data = engineer_features_5min(data)
  total_trades, win_rate, avg_trade_ret, cum_strat_return, trade_dates = (
      run_volatility_proxy_backtest(data, prob_threshold=0.82)
  )

  print('\n--- Volatility Straddle Proxy Diagnostics (Out-of-Sample) ---')
  print(f'Total Option Trades Executed: {int(total_trades):,}')
  print(f'Strategy Win Rate (Straddle Profitability): {win_rate * 100:.2f}%')
  print(f'Average Net Return per Trade: {avg_trade_ret * 100:.4f}%')
  print(f'Cumulative Strategy Multiplier: {cum_strat_return:.2f}x')

  # --- Month-Wise & Year-Wise Trade Distribution Pivot Table ---
  df_trades = pd.DataFrame({'trade_date': pd.to_datetime(trade_dates)})
  df_trades['Year'] = df_trades['trade_date'].dt.year
  df_trades['Month'] = df_trades['trade_date'].dt.month_name()

  month_order = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December',
  ]
  pivot_table = df_trades.pivot_table(
      index='Year',
      columns='Month',
      values='trade_date',
      aggfunc='count',
      fill_value=0,
  )

  existing_months = [m for m in month_order if m in pivot_table.columns]
  pivot_table = pivot_table[existing_months]

  print('\n--- Trade Distribution Heatmap (Year vs. Month) ---')
  print(pivot_table.to_string())
