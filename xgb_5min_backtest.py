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

  # Filter market hours and resample to 5-minute OHLCV
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

  print(
      f'Resampled successfully. Total 5-minute bars:'
      f' {len(df_5min):,}'
  )
  return df_5min


def engineer_features_5min(df):
  print('Engineering 5-minute features and target labels...')
  
  # Returns over 1, 3, and 6 bars (5m, 15m, 30m equivalent)
  df['nifty_ret_1'] = df['close'].pct_change(1)
  df['nifty_ret_3'] = df['close'].pct_change(3)
  df['nifty_ret_6'] = df['close'].pct_change(6)

  df['realized_vol'] = df['nifty_ret_1'].rolling(window=12).std()
  df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
  df['dist_from_ema'] = (df['close'] - df['ema_50']) / df['ema_50']
  df['minutes_from_open'] = (df.index.hour - 9) * 60 + (df.index.minute - 15)

  # ATR (14-period on 5-min bars)
  high_low = df['high'] - df['low']
  high_close = np.abs(df['high'] - df['close'].shift())
  low_close = np.abs(df['low'] - df['close'].shift())
  df['atr'] = (
      pd.concat([high_low, high_close, low_close], axis=1)
      .max(axis=1)
      .rolling(14)
      .mean()
  )

  # Forward range over next 12 bars (12 * 5min = 60 minutes ahead)
  forward_window = 12
  fwd_high = df['high'].shift(-1).rolling(window=forward_window).max()
  fwd_low = df['low'].shift(-1).rolling(window=forward_window).min()
  df['fwd_range'] = fwd_high - fwd_low

  return df.dropna()


def run_5min_ml_backtest(df, prob_threshold=0.75, cost_bps=0.0002):
  print(
      'Training model on 5-minute data and running long/short probability'
      ' backtest...'
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

  # Strict In-Sample Threshold Calculation (Zero Leakage)
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

  print(f'Running backtest on {len(test_df):,} out-of-sample 5-minute bars...')
  start_time = time.time()

  test_df['trade_date'] = test_df.index.date
  all_positions = []
  cutoff_time = pd.to_datetime('15:00:00').time()  # Square off earlier on 5m

  for date, group in test_df.groupby('trade_date'):
    times = group.index.time
    opens = group['open'].values
    lows = group['low'].values
    highs = group['high'].values
    closes = group['close'].values
    emas = group['ema_50'].values
    probs = group['expansion_prob'].values
    atrs = group['atr'].values

    pos = 0
    active = False
    ep = 0.0
    sl = 0.0
    tp = 0.0
    trade_dir = 0

    day_pos = np.zeros(len(group), dtype=int)

    for i in range(len(group) - 1):
      if times[i] >= cutoff_time:
        active = False
        pos = 0
        day_pos[i] = 0
        continue

      cur_low = lows[i]
      cur_high = highs[i]
      cur_close = closes[i]
      cur_ema = emas[i]
      cur_prob = probs[i]
      cur_atr = atrs[i]

      if np.isnan(cur_atr) or cur_atr <= 0:
        day_pos[i] = 0
        continue

      if not active:
        if cur_prob >= prob_threshold:
          active = True
          ep = opens[i + 1]  # Next-bar open execution

          if cur_close > cur_ema:
            trade_dir = 1
            sl = ep - (1.5 * cur_atr)
            tp = ep + (2.5 * cur_atr)
            pos = 1
          else:
            trade_dir = -1
            sl = ep + (1.5 * cur_atr)
            tp = ep - (2.5 * cur_atr)
            pos = -1
      else:
        if trade_dir == 1:
          if cur_low <= sl or cur_high >= tp:
            active = False
            pos = 0
          else:
            pos = 1
        elif trade_dir == -1:
          if cur_high >= sl or cur_low <= tp:
            active = False
            pos = 0
          else:
            pos = -1

      day_pos[i] = pos if active else 0

    day_pos[-1] = 0
    all_positions.extend(day_pos)

  test_df['position'] = all_positions

  # Performance metrics
  test_df['intraday_market_ret'] = test_df.groupby('trade_date')[
      'close'
  ].pct_change().fillna(0)
  test_df['strategy_ret'] = (
      test_df['position'].shift(1) * test_df['intraday_market_ret']
  )

  test_df['trade_occurred'] = test_df['position'].diff().abs().fillna(0)
  test_df.loc[test_df.groupby('trade_date').head(1).index, 'trade_occurred'] = 0

  test_df['strategy_net_ret'] = test_df['strategy_ret'] - (
      test_df['trade_occurred'] * cost_bps
  )

  test_df['cum_market'] = (1 + test_df['intraday_market_ret']).cumprod()
  test_df['cum_strategy'] = (1 + test_df['strategy_net_ret'].fillna(0)).cumprod()

  print(f'Backtest completed in {time.time() - start_time:.2f} seconds.')
  return test_df


if __name__ == '__main__':
  nifty_file = 'nifty_10yr.csv'
  data = load_and_resample_data(nifty_file)
  data = engineer_features_5min(data)
  results = run_5min_ml_backtest(data, prob_threshold=0.75)

  total_trades = results['trade_occurred'].sum()
  winning_trades = results[results['strategy_net_ret'] > 0]
  losing_trades = results[results['strategy_net_ret'] < 0]
  win_rate = (
      len(winning_trades) / (len(winning_trades) + len(losing_trades))
      if (len(winning_trades) + len(losing_trades)) > 0
      else 0
  )

  print('\n--- 5-Minute ML Strategy Diagnostics (Out-of-Sample) ---')
  print(f'Test Bars Processed: {len(results):,}')
  print(f'Total Trades Executed: {int(total_trades):,}')
  print(f'Strategy Win Rate: {win_rate * 100:.2f}%')
  print(
      'Final Market Cumulative Return (Test Period):'
      f' {results["cum_market"].iloc[-1]:.2f}x'
  )
  print(
      'Final Strategy Cumulative Return (Test Period):'
      f' {results["cum_strategy"].iloc[-1]:.2f}x'
  )
