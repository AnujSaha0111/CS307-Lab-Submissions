import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

tickers = ['SPY', 'AAPL', 'TSLA']
start_date = '2015-09-08'
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

def get_processed_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    print(f"Columns for {ticker}: {data.columns}")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [f"{col[0]}_{ticker}" for col in data.columns]
    price_col = f"Close_{ticker}" 
    
    data['Returns'] = data[price_col].pct_change().dropna()
    returns = data['Returns'].values.reshape(-1, 1)
    
    if np.any(np.isnan(returns)):
        print(f"NaN values found in {ticker} returns, dropping them.")
        returns = returns[~np.isnan(returns).any(axis=1)]
    
    return returns

models = {}
hidden_states = {}
for ticker in tickers:
    try:
        returns = get_processed_data(ticker)
        if len(returns) == 0:
            print(f"No valid data for {ticker} after preprocessing.")
            continue
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        model.fit(returns)
        hidden_states[ticker] = model.predict(returns)
        models[ticker] = model
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col[0]}_{ticker}" for col in data.columns]
        pd.DataFrame({'Date': data.index[1:], 'Hidden_State': hidden_states[ticker]}).to_csv(f'{ticker}_states.csv')
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

for ticker in tickers:
    if ticker in models:
        print(f"\n{ticker} HMM Results:")
        for i in range(3):
            state_mask = hidden_states[ticker] == i
            mean = np.mean(models[ticker].means_[i])
            var = np.var(models[ticker].means_[i])
            print(f"State {i+1}: Mean Return = {mean:.6f}, Variance = {var:.6f}")
        print("Transition Matrix:")
        print(models[ticker].transmat_)

plt.figure(figsize=(15, 10))

for idx, ticker in enumerate(tickers, 1):
    if ticker in hidden_states:
        plt.subplot(3, 1, idx)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col[0]}_{ticker}" for col in data.columns]
        price_col = f"Close_{ticker}"
        for i in range(3):
            mask = hidden_states[ticker] == i
            plt.plot(data.index[1:][mask], data[price_col][1:][mask], '.', label=f'State {i+1}')
        plt.title(f'{ticker} Price with 3-State HMM')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('bonus_comparison.png')
plt.show()

with open('transition_matrices.txt', 'w') as f:
    for ticker in tickers:
        if ticker in models:
            f.write(f"{ticker} Transition Matrix:\n{models[ticker].transmat_}\n\n")