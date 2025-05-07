import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Assume these functions are defined in your MVO code cell (or imported):
def calculate_statistics(data):
    returns = data[["SP500_Return", "USBIG_Return"]]
    expected_returns = returns.mean().values
    covariance_matrix = returns.cov().values
    return expected_returns, covariance_matrix

def optimize_portfolio(expected_returns, covariance_matrix, risk_aversion):
    num_assets = len(expected_returns)
    
    def objective_function(weights, returns, cov_matrix, risk_aversion):
        port_return = np.dot(weights, returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        # Negative utility: -(return - risk_aversion*variance)
        return -(port_return - risk_aversion * port_variance)
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * num_assets
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
# Inverse Optimization: Estimate Risk Aversion
# ---------------------
def estimate_risk_aversion(observed_weights, expected_returns, covariance_matrix, candidate_range):
    errors = []
    candidate_risk = []
    best_error = np.inf
    best_risk = None
    best_weights = None

    for risk_aversion in candidate_range:
        try:
            model_weights = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion)
            error = np.linalg.norm(model_weights - observed_weights)
            errors.append(error)
            candidate_risk.append(risk_aversion)
            if error < best_error:
                best_error = error
                best_risk = risk_aversion
                best_weights = model_weights
        except Exception as e:
            print(f"Risk aversion candidate {risk_aversion} failed: {e}")
            continue

    return best_risk, best_weights, candidate_risk, errors

# ---------------------
# Load Synthetic Mutual Fund Data 
# ---------------------
def load_synthetic_mutual_fund_data(file_path="/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed/MutualFunds/mutual_fund_allocations.csv"):
    """
    Load the synthetic mutual fund allocation data
    """
    try:
        # Load the synthetic mutual fund allocations
        synth_data = pd.read_csv(file_path)
        
        # Verify the data has the expected columns
        required_cols = [col for col in synth_data.columns if '_stock' in col or '_bond' in col]
        if not required_cols:
            raise ValueError(f"Synthetic data file does not contain expected columns (_stock, _bond)")
            
        print(f"Successfully loaded synthetic mutual fund data with {len(synth_data)} time periods")
        return synth_data
    except Exception as e:
        print(f"Error loading synthetic mutual fund data: {e}")
        return None

# ---------------------
# Process Synthetic Data to Extract Observed Weights
# ---------------------
def extract_observed_weights(synth_data, date=None):
    """
    Extract average weights across funds for a specific date or for the entire period
    """
    # If date is provided, filter to that date
    if date is not None:
        if 'Date' in synth_data.columns:
            synth_data = synth_data[synth_data['Date'] == date]
            if len(synth_data) == 0:
                print(f"No data found for date {date}, using all data")
                synth_data = synth_data  # Reset to full dataset
    
    # Get lists of stock and bond columns
    stock_cols = [col for col in synth_data.columns if '_stock' in col]
    bond_cols = [col for col in synth_data.columns if '_bond' in col]
    
    # Calculate average weights across all funds and time periods
    avg_stock_weight = synth_data[stock_cols].mean().mean()
    avg_bond_weight = synth_data[bond_cols].mean().mean()
    
    # Ensure weights sum to 1
    total = avg_stock_weight + avg_bond_weight
    avg_stock_weight /= total
    avg_bond_weight /= total
    
    print(f"Observed Average Weights (stocks, bonds): [{avg_stock_weight:.4f}, {avg_bond_weight:.4f}]")
    
    # Return as numpy array for the optimization function
    return np.array([avg_stock_weight, avg_bond_weight])

# ---------------------
# Load and Process Financial Data
# ---------------------
def load_financial_data(split_date="2020-01-01"):
    """
    Load and preprocess financial data for training
    """
    # Define data folder
    data_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed"
    
    # Define file paths
    sp500_file = os.path.join(data_folder, "processed_sp500_data.csv")
    bond_file = os.path.join(data_folder, "processed_bond_data.csv")
    risk_free_file = os.path.join(data_folder, "processed_risk_free_rate_data.csv")
    
    # Load datasets
    sp500_data = pd.read_csv(sp500_file)
    bond_data = pd.read_csv(bond_file)
    risk_free_rate_data = pd.read_csv(risk_free_file)
    
    # Convert date columns to datetime
    sp500_data["Date"] = pd.to_datetime(sp500_data["Date"])
    bond_data["Date"] = pd.to_datetime(bond_data["Date"])
    risk_free_rate_data["Date"] = pd.to_datetime(risk_free_rate_data["Date"])
    
    # Merge datasets
    financial_data = pd.merge(sp500_data[["Date", "SP500_Close"]], 
                            bond_data[["Date", "BAMLC0A0CMEY"]],
                            on="Date", how="inner")
    financial_data = pd.merge(financial_data, 
                            risk_free_rate_data[["Date", "DGS1MO"]],
                            on="Date", how="inner")
    
    # Rename columns for clarity
    financial_data.rename(columns={
        "SP500_Close": "SP500", 
        "BAMLC0A0CMEY": "USBIG", 
        "DGS1MO": "RiskFreeRate"
    }, inplace=True)
    
    # Calculate daily returns
    financial_data["SP500_Return"] = financial_data["SP500"].pct_change()
    financial_data["USBIG_Return"] = financial_data["USBIG"].pct_change()
    
    # Drop rows with NaN values
    financial_data.dropna(inplace=True)
    
    # Split data into train and test sets
    train_data = financial_data[financial_data["Date"] < split_date]
    test_data = financial_data[financial_data["Date"] >= split_date]
    
    return train_data, test_data

# ---------------------
# Main Execution
# ---------------------
# Load the synthetic mutual fund data
synth_data = load_synthetic_mutual_fund_data()

# Extract observed weights from synthetic data
observed_weights = extract_observed_weights(synth_data)
print("Observed Average Mutual Fund Weights (stocks, bonds):", observed_weights)

# Load financial data and split into train/test
train_data, test_data = load_financial_data()

# Calculate expected returns and covariance matrix from training data
train_expected_returns, train_covariance_matrix = calculate_statistics(train_data)

# ---------------------
# Estimate Risk Aversion via Inverse Optimization
# ---------------------
# Define a candidate risk aversion range (for example, from 0.1 to 10)
candidate_range = np.linspace(0.1, 10, 100)

best_risk, best_model_weights, candidate_risk, errors = estimate_risk_aversion(
    observed_weights, train_expected_returns, train_covariance_matrix, candidate_range
)

print("Estimated Risk Aversion Parameter:", best_risk)
print("Optimal Weights (MVO) at Estimated Risk Aversion:", best_model_weights)

# Optional: Save the candidate risk vs. error data for further analysis
results_df = pd.DataFrame({
    "Risk_Aversion": candidate_risk,
    "Error": errors
})
results_df.to_csv("inverse_optimization_results.csv", index=False)
print("Inverse optimization results saved to 'inverse_optimization_results.csv'.")