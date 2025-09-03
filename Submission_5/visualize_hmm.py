import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_regimes(data, ticker, n_states, price_col, output_file='stock_regimes.png'):
    """
    Plot stock prices with color-coded hidden states.
    
    Parameters:
    - data (pandas.DataFrame): Data with price column and 'Hidden_State' columns
    - ticker (str): Stock ticker symbol
    - n_states (int): Number of hidden states
    - price_col (str): Name of the price column
    - output_file (str): Path to save the plot
    """
    plt.figure(figsize=(15, 7))
    colors = ['blue', 'red', 'green'][:n_states]
    for i in range(n_states):
        mask = data['Hidden_State'] == i
        plt.plot(data.index[mask], data[price_col][mask], '.',
                 label=f'State {i+1}', color=colors[i], alpha=0.6)
    plt.title(f'{ticker} Stock Prices with HMM States')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Stock regimes plot saved to {output_file}")
    plt.show()

def plot_returns_distribution(data, n_states, output_file='returns_distribution.png'):
    """
    Plot distribution of returns for each hidden state.
    
    Parameters:
    - data (pandas.DataFrame): Data with 'Returns' and 'Hidden_State' columns
    - n_states (int): Number of hidden states
    - output_file (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green'][:n_states]
    for i in range(n_states):
        mask = data['Hidden_State'] == i
        sns.kdeplot(data['Returns'][mask], label=f'State {i+1}', color=colors[i], fill=True)
    plt.title('Distribution of Returns by Hidden State')
    plt.xlabel('Daily Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_file)
    print(f"Returns distribution plot saved to {output_file}")
    plt.show()

def main():
    try:
        data = pd.read_csv('data_with_states.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        print("Columns in data with states:", data.columns)
    except FileNotFoundError:
        print("Error: data_with_states.csv not found. Run fit_hmm.py first.")
        return
    
    possible_price_cols = ['Adj Close', 'Adjusted Close', 'adj_close', 'Close']
    price_col = None
    for col in possible_price_cols:
        if col in data.columns:
            price_col = col
            break
    if price_col is None:
        print("Error: No price column found. Available columns:", list(data.columns))
        return
    
    ticker = "SPY"
    n_states = 2
    
    plot_regimes(data, ticker, n_states, price_col)
    plot_returns_distribution(data, n_states)

if __name__ == "__main__":
    main()