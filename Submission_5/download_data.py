import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'SPY')
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    - pandas.DataFrame: Historical stock data
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]
        
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            if col in stock_data.columns:
                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
        
        print("Columns in downloaded data:", stock_data.columns)
        stock_data.to_csv(f'{ticker}_historical_data.csv')
        print(f"Data for {ticker} downloaded and saved to {ticker}_historical_data.csv")
        return stock_data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def main():
    ticker = "SPY" 
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        print(f"Downloaded data shape: {stock_data.shape}")
        print(stock_data.head())

if __name__ == "__main__":
    main()