import pandas as pd
import numpy as np

def preprocess_data(stock_data, output_file='preprocessed_data.csv'):
    """
    Preprocess stock data by calculating daily returns and handling missing values.
    
    Parameters:
    - stock_data (pandas.DataFrame): Historical stock data with 'Adj Close' or 'Close' column
    - output_file (str): Path to save preprocessed data
    
    Returns:
    - pandas.DataFrame: Preprocessed data with returns
    """
    try:
        possible_price_cols = ['Adj Close', 'Adjusted Close', 'adj_close', 'Close']
        price_col = None
        for col in possible_price_cols:
            if col in stock_data.columns:
                price_col = col
                break
        if price_col is None:
            raise KeyError("No price column (Adj Close or Close) found. Available columns: " + str(list(stock_data.columns)))
        
        stock_data[price_col] = pd.to_numeric(stock_data[price_col], errors='coerce')
        
        stock_data['Returns'] = stock_data[price_col].pct_change(fill_method=None)
        
        stock_data = stock_data.replace([np.inf, -np.inf], np.nan)
        initial_rows = stock_data.shape[0]
        stock_data = stock_data.dropna()
        print(f"Removed {initial_rows - stock_data.shape[0]} rows with missing/invalid data")
        
        stock_data.to_csv(output_file)
        print(f"Preprocessed data saved to {output_file}")
        return stock_data
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def main():
    ticker = "SPY"
    try:
        stock_data = pd.read_csv(f'{ticker}_historical_data.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        print("Columns in CSV:", stock_data.columns)
    except FileNotFoundError:
        print(f"Error: {ticker}_historical_data.csv not found. Run download_data.py first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    preprocessed_data = preprocess_data(stock_data)
    if preprocessed_data is not None:
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
        print(preprocessed_data[[price_col, 'Returns']].head() if 'price_col' in locals() else preprocessed_data.head())

if __name__ == "__main__":
    main()