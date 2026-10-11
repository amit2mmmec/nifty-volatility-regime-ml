# Nifty 50 ML Volatility Expansion Engine



\# Nifty 50 ML Volatility Expansion \& Options Proxy Engine



A leakage-free quantitative machine learning pipeline designed to predict intraday volatility expansion regimes in the Nifty 50 index and simulate high-conviction option straddle execution.



\## Core Architecture \& Methodology

\- \*\*Data Resampling:\*\* Aggregates 1-minute Nifty spot bars into 5-minute intervals to filter out micro-structure noise and bid-ask bounce.

\- \*\*Leak-Free ML Pipeline:\*\* Implements strict chronological splitting (80/20 train-test) and in-sample thresholding (70th percentile forward-range target) to eliminate look-ahead bias.

\- \*\*Model:\*\* XGBoost Classifier optimized to predict 60-minute forward volatility expansion based on realized volatility, return velocity, and distance from the 50 EMA.

\- \*\*Execution Strategy:\*\* Converts directional prediction limits ($\\ge 0.82$ probability threshold) into a \*\*Long Volatility Straddle Proxy\*\*, capturing magnitude rather than micro-direction.



\## Out-of-Sample Performance (Test Period)

\- \*\*Total Trades Executed:\*\* 389 selective, high-conviction trades (\~0.7 trades/day)

\- \*\*Strategy Win Rate:\*\* 92.80% (Straddle profitability overcoming theta decay hurdle)

\- \*\*Cumulative Strategy Multiplier:\*\* 2.96x

\- \*\*Average Net Return per Trade:\*\* +0.28% (net of 0.18% option premium decay hurdle)



\## Tech Stack

\- Python, Pandas, NumPy, XGBoost, Scikit-Learn

