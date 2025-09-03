import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import pickle

def fit_hmm_model(returns, n_states=2, output_model_file='hmm_model.pkl'):
    """
    Fit a Gaussian HMM to the returns data.
    
    Parameters:
    - returns (pandas.Series): Daily returns
    - n_states (int): Number of hidden states
    - output_model_file (str): Path to save the trained model
    
    Returns:
    - model: Trained GaussianHMM model
    - hidden_states: Predicted hidden states
    """
    try:
        X = returns.values.reshape(-1, 1)
        
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X)
        
        hidden_states = model.predict(X)

        with open(output_model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"HMM model saved to {output_model_file}")
        
        print("\nHidden State Parameters:")
        means = model.means_.flatten()
        variances = np.sqrt(np.array([np.diag(model.covars_[i]) for i in range(model.n_components)]).flatten())
        for i in range(model.n_components):
            print(f"State {i+1}: Mean Return = {means[i]:.6f}, Variance = {variances[i]:.6f}")
        
        print("\nTransition Matrix:")
        print(model.transmat_)
        
        return model, hidden_states
    except Exception as e:
        print(f"Error fitting HMM: {e}")
        return None, None

def main():
    try:
        data = pd.read_csv('preprocessed_data.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        print("Columns in preprocessed data:", data.columns)
    except FileNotFoundError:
        print("Error: preprocessed_data.csv not found. Run preprocess_data.py first.")
        return
    
    returns = data['Returns']
    n_states = 2 
    
    model, hidden_states = fit_hmm_model(returns, n_states)
    if model is not None:
        possible_price_cols = ['Adj Close', 'Adjusted Close', 'adj_close', 'Close']
        price_col = None
        for col in possible_price_cols:
            if col in data.columns:
                price_col = col
                break
        if price_col is None:
            print("Error: No price column found. Available columns:", list(data.columns))
            return
        
        data['Hidden_State'] = hidden_states
        data.to_csv('data_with_states.csv')
        print("Data with hidden states saved to data_with_states.csv")

if __name__ == "__main__":
    main()