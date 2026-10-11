import time
import numpy as np
import pandas as pd


def load_and_prepare_data(nifty_path):
  print('Loading Nifty dataset...')
  df = pd.read_csv(nifty_path)
  df['date'] = pd.to_datetime(df['date'])
  df.set_index('date', inplace=True)
  df.columns = [c.lower() for c in df.columns]

  # Filter standard market hours
  df = df.between_time('09:15', '15:29')
  return df


anti_lookahead_msg = (
    'Calculating daily pivots, 50 EMA, and 14-period ATR (Zero Lookahead'
    ' Bias)...'
)


def calculate_indicators(df):
  print(anti_lookahead_msg)
  df['trade_date'] = df.index.date

  # 1. Daily Pivots (Shifted by 1 day to ensure no lookahead bias)
  daily = (
      df.groupby('trade_date')
      .agg({'high': 'max', 'low': 'min', 'close': 'last'})
      .shift(1)
  )

  daily['pp'] = (daily['high'] + daily['low'] + daily['close']) / 3
  daily['bc'] = (daily['high'] + daily['low']) / 2
  daily['tc'] = (2 * daily['pp']) - daily['bc']
  daily['s1'] = (2 * daily['pp']) - daily['high']
  daily['r1'] = (2 * daily['pp']) - daily['low']
  daily = daily.dropna()

  df = df.reset_index().merge(
      daily[['pp', 'bc', 'tc', 's1', 'r1']], on='trade_date', how='inner'
  )
  df.set_index('date', inplace=True)

  # 2. Intraday Trend Filter (50 EMA)
  df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

  # 3. Average True Range (ATR 14) for dynamic stops/targets
  high_low = df['high'] - df['low']
  high_close = np.abs(df['high'] - df['close'].shift())
  low_close = np.abs(df['low'] - df['close'].shift())
  true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
  df['atr'] = true_range.rolling(window=14).mean()

  return df


def run_atr_pivot_backtest(df, cost_bps=0.0002):
  print(
      'Running ATR-Scaled Pivot Backtest with Next-Bar Open Execution (Zero'
      ' Lookahead)...'
  )
  start_time = time.time()

  df['trade_date'] = df.index.date
  all_positions = []
  cutoff_time = pd.to_datetime('15:15:00').time()

  for date, group in df.groupby('trade_date'):
    times = group.index.time
    opens = group['open'].values
    lows = group['low'].values
    highs = group['high'].values
    closes = group['close'].values
    ema50 = group['ema_50'].values
    atrs = group['atr'].values

    s1 = group['s1'].iloc[0]
    bc = group['bc'].iloc[0]
    support_level = min(s1, bc)

    pos = 0
    active = False
    ep = 0.0
    sl_price = 0.0
    tp_price = 0.0

    day_pos = np.zeros(len(group), dtype=int)

    for i in range(len(group) - 1):  # leave room for next-bar execution
      if times[i] >= cutoff_time:
        active = False
        pos = 0
        day_pos[i] = 0
        continue

      cur_low = lows[i]
      cur_high = highs[i]
      cur_close = closes[i]
      cur_ema = ema50[i]
      cur_atr = atrs[i]

      if np.isnan(cur_atr) or cur_atr <= 0:
        day_pos[i] = 0
        continue

      if not active:
        # Signal condition met at bar i: Price dips near support, closes above support, and above 50 EMA
        if (
            (cur_low <= support_level * (1 + 0.0005))
            and (cur_close > support_level)
            and (cur_close > cur_ema)
        ):
          # Execute trade at the OPEN of the NEXT bar (i+1) to eliminate lookahead bias
          active = True
          ep = opens[i + 1]
          sl_price = ep - (1.5 * cur_atr)
          tp_price = ep + (2.5 * cur_atr)
          pos = 1
      else:
        # Manage active trade using bar i+1 price action (or current bar if evaluating intra-bar)
        # Check if Stop Loss or Profit Target was breached during bar i
        if cur_low <= sl_price or cur_high >= tp_price:
          active = False
          pos = 0
        else:
          pos = 1

      day_pos[i] = pos if active else 0

    # Ensure last bar is squared off
    day_pos[-1] = 0
    all_positions.extend(day_pos)

  df['position'] = all_positions

  # Performance Metrics
  df['intraday_market_ret'] = df.groupby('trade_date')[
      'close'
  ].pct_change().fillna(0)
  df['strategy_ret'] = df['position'].shift(1) * df['intraday_market_ret']

  df['trade_occurred'] = df['position'].diff().abs().fillna(0)
  df.loc[df.groupby('trade_date').head(1).index, 'trade_occurred'] = 0

  df['strategy_net_ret'] = (
      df['strategy_ret'] - (df['trade_occurred'] * cost_bps)
  )

  df['cum_market'] = (1 + df['intraday_market_ret']).cumprod()
  df['cum_strategy'] = (1 + df['strategy_net_ret'].fillna(0)).cumprod()

  print(f'Backtest completed in {time.time() - start_time:.2f} seconds.')
  return df


if __name__ == '__main__':
  nifty_file = 'nifty_10yr.csv'

  data = load_and_prepare_data(nifty_file)
  data = calculate_indicators(data)
  results = run_atr_pivot_backtest(data)

  total_trades = results['trade_occurred'].sum()
  winning_trades = results[results['strategy_net_ret'] > 0]
  losing_trades = results[results['strategy_net_ret'] < 0]
  win_rate = (
      len(winning_trades) / (len(winning_trades) + len(losing_trades))
      if (len(winning_trades) + len(losing_trades)) > 0
      else 0
  )

  print('\n--- ATR-Scaled Pivot Backtest Diagnostics ---')
  print(f'Total Minutes Processed: {len(results):,}')
  print(f'Total Trades Executed: {int(total_trades):,}')
  print(f'Strategy Win Rate: {win_rate * 100:.2f}%')
  print(f'Final Market Cumulative Return: {results["cum_market"].iloc[-1]:.2f}x')
  print(
      'Final Strategy Cumulative Return:'
      f' {results["cum_strategy"].iloc[-1]:.2f}x'
  )
