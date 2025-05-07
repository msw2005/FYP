import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

# ---------------------
# Configuration
# ---------------------
data_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed"
damodaran_data_path = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Damodaran- Expected Returns/histretSP.xlsx"
annualization_factor = 252  # For daily returns (choose 252 trading days) if annualization is desired

# ---------------------
# Data Loading
# ---------------------
def load_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Failed to load file {filepath}: {e}")

sp500_file = os.path.join(data_folder, "processed_sp500_data.csv")
bond_file = os.path.join(data_folder, "processed_bond_data.csv")
risk_free_file = os.path.join(data_folder, "processed_risk_free_rate_data.csv")

sp500_data = load_csv(sp500_file)
bond_data = load_csv(bond_file)
risk_free_rate_data = load_csv(risk_free_file)
damodaran_data = pd.read_excel(damodaran_data_path)  # Not used in current model, but loaded for extension

# ---------------------
# Data Preprocessing
# ---------------------
# Convert 'Date' columns to datetime
for df in [sp500_data, bond_data, risk_free_rate_data]:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Merge datasets on 'Date'
try:
    financial_data = pd.merge(sp500_data[["Date", "SP500_Close"]], 
                              bond_data[["Date", "BAMLC0A0CMEY"]],
                              on="Date", how="inner")
    financial_data = pd.merge(financial_data, 
                              risk_free_rate_data[["Date", "DGS1MO"]],
                              on="Date", how="inner")
except Exception as e:
    raise Exception("Error while merging datasets: " + str(e))

# Rename columns for clarity
financial_data.rename(columns={
    "SP500_Close": "SP500", 
    "BAMLC0A0CMEY": "USBIG", 
    "DGS1MO": "RiskFreeRate"
}, inplace=True)

# Handle missing values
financial_data = financial_data.ffill().bfill()

# Calculate daily returns
financial_data["SP500_Return"] = financial_data["SP500"].pct_change()
financial_data["USBIG_Return"] = financial_data["USBIG"].pct_change()

# Drop remaining NaN values
financial_data.dropna(inplace=True)

# ---------------------
# Split Data into Training and Testing Sets
# ---------------------
train_data = financial_data[financial_data["Date"] < "2020-01-01"]
test_data = financial_data[financial_data["Date"] >= "2020-01-01"]

# ---------------------
# Statistics Calculation
# ---------------------
def calculate_statistics(data):
    returns = data[["SP500_Return", "USBIG_Return"]]
    expected_returns = returns.mean().values
    covariance_matrix = returns.cov().values
    return expected_returns, covariance_matrix

# ---------------------
# Portfolio Optimization Function
# ---------------------
def optimize_portfolio(expected_returns, covariance_matrix, risk_aversion):
    num_assets = len(expected_returns)
    
    def objective_function(weights, returns, cov_matrix, risk_aversion):
        # Calculate portfolio return and risk (variance)
        port_return = np.dot(weights, returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        # We minimize the negative utility: -(return - risk_aversion * variance)
        return -(port_return - risk_aversion * port_variance)
    
    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    
    # Bounds: weights between 0 and 1 (no short selling)
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Initial guess: equal weights
    initial_weights = np.ones(num_assets) / num_assets
    
    result = minimize(
        objective_function,
        x0=initial_weights,
        args=(expected_returns, covariance_matrix, risk_aversion),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)

# ---------------------
# Portfolio Performance Evaluation
# ---------------------
def evaluate_portfolio(weights, expected_returns, covariance_matrix, risk_free_rate, annualize=False):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    
    if annualize:
        portfolio_return *= annualization_factor
        portfolio_volatility *= np.sqrt(annualization_factor)
    
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return {
        "Portfolio Return": portfolio_return,
        "Portfolio Volatility": portfolio_volatility,
        "Sharpe Ratio": sharpe_ratio
    }

# ---------------------
# Main Execution for Training and Testing
# ---------------------
if __name__ == "__main__":
    # Example: using a fixed risk aversion parameter.
    risk_aversion = 3
    
    # Training set optimization and evaluation
    try:
        train_expected_returns, train_covariance_matrix = calculate_statistics(train_data)
        optimal_weights_train = optimize_portfolio(train_expected_returns, train_covariance_matrix, risk_aversion)
        print("Optimal Weights (Training Set):", optimal_weights_train)
        
        # Note: risk_free_rate provided as a percentage; hence division by 100.
        avg_rf_train = train_data["RiskFreeRate"].mean() / 100  
        train_performance = evaluate_portfolio(optimal_weights_train, train_expected_returns, train_covariance_matrix, avg_rf_train, annualize=True)
        print("Performance (Training Set):", train_performance)
    except Exception as e:
        print("Optimization failed (Training Set):", e)
    
    # Testing set optimization and evaluation
    try:
        test_expected_returns, test_covariance_matrix = calculate_statistics(test_data)
        optimal_weights_test = optimize_portfolio(test_expected_returns, test_covariance_matrix, risk_aversion)
        print("Optimal Weights (Testing Set):", optimal_weights_test)
        
        avg_rf_test = test_data["RiskFreeRate"].mean() / 100  
        test_performance = evaluate_portfolio(optimal_weights_test, test_expected_returns, test_covariance_matrix, avg_rf_test, annualize=True)
        print("Performance (Testing Set):", test_performance)
    except Exception as e:
        print("Optimization failed (Testing Set):", e)
