import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# ---------------------
# Configuration
# ---------------------
data_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed"
results_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Results"

# Ensure results directory exists
os.makedirs(results_folder, exist_ok=True)

# ---------------------
# Data Loading and Utility Functions
# ---------------------
def load_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Failed to load file {filepath}: {e}")

def calculate_statistics(data):
    returns = data[["SP500_Return", "USBIG_Return"]]
    expected_returns = returns.mean().values
    covariance_matrix = returns.cov().values
    return expected_returns, covariance_matrix

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

def evaluate_portfolio(weights, returns_data, risk_free_rate=0.0):
    """
    Evaluate portfolio performance using historical returns
    
    Args:
        weights: Portfolio weights [stock_weight, bond_weight]
        returns_data: DataFrame with columns for stock and bond returns
        risk_free_rate: Risk-free rate (annual)
    
    Returns:
        dict: Performance metrics
    """
    # Calculate portfolio returns
    portfolio_returns = weights[0] * returns_data["SP500_Return"] + weights[1] * returns_data["USBIG_Return"]
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    # Calculate metrics
    annualized_return = ((1 + cumulative_returns.iloc[-1]) ** (252 / len(returns_data))) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_values = (1 + portfolio_returns).cumprod()
    running_max = cumulative_values.cummax()
    drawdown = (cumulative_values - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    return {
        "portfolio_returns": portfolio_returns,
        "cumulative_returns": cumulative_returns,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

def calculate_max_drawdown(portfolio_values):
    """Calculate the maximum drawdown of a portfolio"""
    portfolio_values = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdown)
    return abs(max_drawdown)

# ---------------------
# Main Function
# ---------------------
def main():
    # Load datasets
    sp500_data = load_csv(os.path.join(data_folder, "processed_sp500_data.csv"))
    bond_data = load_csv(os.path.join(data_folder, "processed_bond_data.csv"))
    risk_free_rate_data = load_csv(os.path.join(data_folder, "processed_risk_free_rate_data.csv"))
    
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
    
    # Handle missing values
    financial_data = financial_data.ffill().bfill()
    
    # Calculate daily returns
    financial_data["SP500_Return"] = financial_data["SP500"].pct_change()
    financial_data["USBIG_Return"] = financial_data["USBIG"].pct_change()
    
    # Clip extreme returns
    financial_data["SP500_Return"] = financial_data["SP500_Return"].clip(-0.1, 0.1)
    financial_data["USBIG_Return"] = financial_data["USBIG_Return"].clip(-0.1, 0.1)
    
    # Drop rows with NaN values
    financial_data.dropna(inplace=True)
    
    # Split data into train and test sets
    split_date = "2020-01-01"
    train_data = financial_data[financial_data["Date"] < split_date].copy()
    test_data = financial_data[financial_data["Date"] >= split_date].copy()
    
    # Load risk aversion parameter or use default
    try:
        inverse_opt_results = pd.read_csv("/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Models/inverse_optimization_results.csv")
        best_idx = inverse_opt_results["Error"].idxmin()
        risk_aversion = inverse_opt_results.iloc[best_idx]["Risk_Aversion"]
    except:
        risk_aversion = 0.1  # Default value
    
    print(f"Using risk aversion parameter: {risk_aversion}")
    
    # Calculate MVO weights using training data
    train_expected_returns, train_covariance_matrix = calculate_statistics(train_data)
    optimal_weights = optimize_portfolio(train_expected_returns, train_covariance_matrix, risk_aversion)
    print(f"MVO Optimal Weights [Stock, Bond]: {optimal_weights}")
    
    # Create benchmark portfolios
    benchmark_weights = {
        "MVO": optimal_weights,
        "Equal Weight": np.array([0.5, 0.5]),
        "Observed Weights": np.array([0.68, 0.32])  # From fund weights
    }
    
    # Evaluate portfolios on test data
    avg_rf_rate = test_data["RiskFreeRate"].mean() / 100  # Convert to decimal
    results = {}
    
    for name, weights in benchmark_weights.items():
        results[name] = evaluate_portfolio(weights, test_data, avg_rf_rate)
        print(f"\n{name} Portfolio Performance:")
        print(f"  Annualized Return: {results[name]['annualized_return']:.4f}")
        print(f"  Volatility: {results[name]['volatility']:.4f}")
        print(f"  Sharpe Ratio: {results[name]['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {results[name]['max_drawdown']:.4f}")
    
    # Visualize portfolio performance
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        if name == "MVO":
            plt.plot(test_data["Date"], (1 + result["cumulative_returns"]), 
                    label=name, linewidth=2)
        elif name == "Equal Weight":
            plt.plot(test_data["Date"], (1 + result["cumulative_returns"]), 
                    label=name, linestyle="--")
        else:
            plt.plot(test_data["Date"], (1 + result["cumulative_returns"]), 
                    label=name, linestyle="-.")
    
    plt.title("Portfolio Performance Comparison", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Return (1 + r)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add a summary table as text
    summary_text = "Performance Summary:\n"
    for name in results:
        summary_text += f"{name}:\n"
        summary_text += f"  Return: {results[name]['annualized_return']:.2%}\n"
        summary_text += f"  Sharpe: {results[name]['sharpe_ratio']:.2f}\n"
    
    plt.figtext(0.15, 0.15, summary_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    plot_path = os.path.join(results_folder, "mvo_performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
    
    plt.show()
    
    # Create more detailed visualizations
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Cumulative Returns
    for name, result in results.items():
        style = '-' if name == 'MVO' else ('--' if name == 'Equal Weight' else '-.')
        axs[0].plot(test_data["Date"], (1 + result["cumulative_returns"]), 
                  label=name, linestyle=style)
    
    axs[0].set_title("Cumulative Portfolio Returns", fontsize=16)
    axs[0].set_ylabel("Cumulative Value (1 + r)", fontsize=14)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=12)
    
    # Plot 2: Return Distribution
    colors = ['blue', 'green', 'orange']
    for i, (name, result) in enumerate(results.items()):
        returns = result["portfolio_returns"]
        axs[1].hist(returns, bins=50, alpha=0.5, label=name, color=colors[i])
        axs[1].axvline(returns.mean(), color=colors[i], linestyle='--', 
                     label=f"{name} Mean: {returns.mean():.4f}")
    
    axs[1].set_title("Return Distribution Comparison", fontsize=16)
    axs[1].set_xlabel("Daily Return", fontsize=14)
    axs[1].set_ylabel("Frequency", fontsize=14)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the detailed figure
    detailed_plot_path = os.path.join(results_folder, "mvo_detailed_comparison.png")
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches="tight")
    print(f"Detailed plot saved to {detailed_plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()