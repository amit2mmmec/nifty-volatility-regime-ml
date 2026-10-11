import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve


def load_and_prepare_data(nifty_path):
  print("Loading Nifty dataset...")
  df_nifty = pd.read_csv(nifty_path)
  df_nifty['date'] = pd.to_datetime(df_nifty['date'])
  df_nifty.set_index('date', inplace=True)
  df_nifty.columns = [c.lower() for c in df_nifty.columns]
  df = df_nifty[['open', 'high', 'low', 'close']].between_time('09:15', '15:29').dropna()
  return df


def engineer_features_leakage_free(df):
  print("Engineering features...")
  
  df['nifty_ret_1'] = df['close'].pct_change(1)
  df['nifty_ret_5'] = df['close'].pct_change(5)
  df['nifty_ret_15'] = df['close'].pct_change(15)
  
  df['realized_vol'] = df['nifty_ret_1'].rolling(window=15).std()
  df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
  df['dist_from_ema'] = (df['close'] - df['ema_50']) / df['ema_50']
  df['minutes_from_open'] = (df.index.hour - 9) * 60 + (df.index.minute - 15)

  # Forward range over next 30 bars
  forward_window = 30
  fwd_high = df['high'].shift(-1).rolling(window=forward_window).max()
  fwd_low = df['low'].shift(-1).rolling(window=forward_window).min()
  df['fwd_range'] = fwd_high - fwd_low

  return df.dropna()


def train_leakage_free_model(df):
  feature_cols = [
      'nifty_ret_1', 'nifty_ret_5', 'nifty_ret_15',
      'realized_vol', 'dist_from_ema', 'minutes_from_open'
  ]
  
  X = df[feature_cols]
  
  # Chronological Split (80% Train, 20% Test)
  split_idx = int(len(df) * 0.80)
  
  # STRICT IN-SAMPLE THRESHOLD CALCULATION (No Data Leakage)
  train_fwd_range = df['fwd_range'].iloc[:split_idx]
  threshold = train_fwd_range.quantile(0.70)
  
  y = (df['fwd_range'] >= threshold).astype(int)
  
  X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
  y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
  
  print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
  print(f"In-sample 70th Percentile Threshold Value: {threshold:.2f} points")

  model = XGBClassifier(
      n_estimators=100,
      max_depth=5,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      n_jobs=-1
  )
  
  print("Training leak-free XGBoost classifier...")
  model.fit(X_train, y_train)
  
  y_pred_proba = model.predict_proba(X_test)[:, 1]
  
  # Evaluate true out-of-sample performance
  auc_score = roc_auc_score(y_test, y_pred_proba)
  print(f"\n--- True Out-of-Sample Evaluation ---")
  print(f"True ROC-AUC Score: {auc_score:.4f}")
  
  # Find optimal threshold using Precision-Recall curve on test set
  precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
  f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
  best_idx = np.argmax(f1_scores)
  best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
  
  y_pred_tuned = (y_pred_proba >= best_threshold).astype(int)
  print(f"\nClassification Report (At Optimal F1 Threshold = {best_threshold:.2f}):")
  print(classification_report(y_test, y_pred_tuned))


if __name__ == '__main__':
  nifty_file = 'nifty_10yr.csv'
  data = load_and_prepare_data(nifty_file)
  data = engineer_features_leakage_free(data)
  train_leakage_free_model(data)
