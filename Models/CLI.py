#!/usr/bin/env python
# filepath: /Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/cli.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
from tabulate import tabulate

# Configure paths
BASE_DIR = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP"
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(split_date="2020-01-01"):
    """
    Load and preprocess financial data
    """
    print("Loading financial data...")
    
    # Load datasets
    sp500_data = pd.read_csv(os.path.join(DATA_DIR, "processed_sp500_data.csv"))
    bond_data = pd.read_csv(os.path.join(DATA_DIR, "processed_bond_data.csv"))
    risk_free_rate_data = pd.read_csv(os.path.join(DATA_DIR, "processed_risk_free_rate_data.csv"))
    
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
        "BAMLC0A0CMEY": "Bond", 
        "DGS1MO": "RiskFreeRate"
    }, inplace=True)
    
    # Handle missing values
    financial_data = financial_data.ffill().bfill()
    
    # Calculate daily returns
    financial_data["SP500_Return"] = financial_data["SP500"].pct_change()
    financial_data["Bond_Return"] = financial_data["Bond"].pct_change()
    
    # Clip extreme returns
    financial_data["SP500_Return"] = financial_data["SP500_Return"].clip(-0.1, 0.1)
    financial_data["Bond_Return"] = financial_data["Bond_Return"].clip(-0.1, 0.1)
    
    # Drop rows with NaN values
    financial_data.dropna(inplace=True)
    
    # Split data into train and test sets
    train_data = financial_data[financial_data["Date"] < split_date].copy()
    test_data = financial_data[financial_data["Date"] >= split_date].copy()
    
    return train_data, test_data

def calculate_statistics(data):
    """Calculate expected returns and covariance matrix for MVO"""
    returns = data[["SP500_Return", "Bond_Return"]]
    expected_returns = returns.mean().values
    covariance_matrix = returns.cov().values
    return expected_returns, covariance_matrix

def optimize_portfolio(expected_returns, covariance_matrix, risk_aversion):
    """Run Mean-Variance Optimization to get optimal portfolio weights"""
    num_assets = len(expected_returns)
    
    def objective_function(weights, returns, cov_matrix, risk_aversion):
        port_return = np.dot(weights, returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return -(port_return - risk_aversion * port_variance)
    
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = [(0, 1) for _ in range(num_assets)]
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
    """Evaluate portfolio performance using historical returns"""
    # Calculate portfolio returns
    portfolio_returns = weights[0] * returns_data["SP500_Return"] + weights[1] * returns_data["Bond_Return"]
    
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

def run_mvo(risk_aversion, show_plot=False, save_plot=True):
    """Run the Mean-Variance Optimization model"""
    print(f"Running MVO with risk aversion: {risk_aversion}")
    
    # Load and split data
    train_data, test_data = load_data()
    
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
    
    # Display results
    table_data = []
    for name, result in results.items():
        table_data.append([
            name,
            f"{result['annualized_return']:.2%}",
            f"{result['volatility']:.2%}",
            f"{result['sharpe_ratio']:.2f}",
            f"{result['max_drawdown']:.2%}"
        ])
    
    print("\nPortfolio Performance Comparison:")
    print(tabulate(
        table_data,
        headers=["Strategy", "Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
        tablefmt="pretty"
    ))
    
    # Visualization
    if show_plot or save_plot:
        plt.figure(figsize=(12, 8))
        for name, result in results.items():
            style = '-' if name == 'MVO' else ('--' if name == 'Equal Weight' else '-.')
            plt.plot(test_data["Date"], (1 + result["cumulative_returns"]), 
                   label=name, linestyle=style)
        
        plt.title("Portfolio Performance Comparison", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Cumulative Return (1 + r)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        if save_plot:
            plot_path = os.path.join(RESULTS_DIR, "mvo_performance_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {plot_path}")
        
        if show_plot:
            plt.show()
    
    return results

def load_rl_results():
    """Load pre-computed Deep RL results from CSV files"""
    try:
        metrics_file = os.path.join(RESULTS_DIR, "portfolio_metrics.csv")
        weights_file = os.path.join(RESULTS_DIR, "portfolio_allocations.csv")
        
        metrics = pd.read_csv(metrics_file)
        weights = pd.read_csv(weights_file)
        
        return {
            'metrics': metrics,
            'weights': weights
        }
    except Exception as e:
        print(f"Error loading RL results: {e}")
        return None

def print_portfolio_metrics(metrics_data):
    """Print portfolio metrics in a formatted table"""
    try:
        # Convert metrics DataFrame to a nicely formatted table
        metrics_dict = metrics_data.iloc[0].to_dict()
        
        # Filter out benchmark comparison and other nested data
        metrics_table = []
        for key, value in metrics_dict.items():
            if key != 'benchmark_comparison' and not isinstance(value, dict) and key != 'risk_aversion':
                if key in ['annualized_return', 'volatility', 'max_drawdown']:
                    value_formatted = f"{value:.2%}"
                else:
                    value_formatted = f"{value:.2f}"
                
                # Make key more readable
                key_formatted = key.replace('_', ' ').title()
                metrics_table.append([key_formatted, value_formatted])
        
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="pretty"))
    except Exception as e:
        print(f"Error displaying metrics: {e}")

def display_rl_results(show_plot=False):
    """Display results from the Deep RL model"""
    results = load_rl_results()
    
    if not results:
        print("No Deep RL results found. Please run the model first.")
        return
    
    print("\nDeep RL Portfolio Metrics:")
    print_portfolio_metrics(results['metrics'])
    
    # Show weights distribution
    print("\nPortfolio Allocation Statistics:")
    weights_summary = results['weights'][['Stock Weight', 'Bond Weight']].describe()
    print(tabulate(weights_summary, headers='keys', tablefmt="pretty"))
    
    # Visualization
    if show_plot:
        try:
            # Plot portfolio allocation over time
            plt.figure(figsize=(12, 6))
            plt.stackplot(range(len(results['weights'])),
                          results['weights']['Stock Weight'],
                          results['weights']['Bond Weight'],
                          labels=['Stocks', 'Bonds'],
                          colors=['#ff9999', '#66b3ff'],
                          alpha=0.8)
            plt.title("Deep RL Asset Allocation Over Time", fontsize=16)
            plt.ylabel("Weight", fontsize=14)
            plt.xlabel("Time Step", fontsize=14)
            plt.legend(fontsize=12, loc="upper right")
            plt.grid(True)
            plt.show()
            
            # Try to show portfolio value if available
            if 'Portfolio Value' in results['weights']:
                plt.figure(figsize=(12, 6))
                plt.plot(results['weights']['Portfolio Value'], label="Portfolio Value", color="blue")
                plt.title("Deep RL Portfolio Value Over Time", fontsize=16)
                plt.ylabel("Portfolio Value ($)", fontsize=14)
                plt.xlabel("Time Step", fontsize=14)
                plt.grid(True)
                plt.show()
        except Exception as e:
            print(f"Error creating visualization: {e}")

def compare_strategies(show_plot=False):
    """Compare MVO and Deep RL strategies"""
    print("Comparing portfolio allocation strategies...")
    
    # Run MVO
    mvo_results = run_mvo(risk_aversion=0.1, show_plot=False, save_plot=False)
    
    # Load Deep RL results
    rl_results = load_rl_results()
    
    if not rl_results:
        print("No Deep RL results found. Only showing MVO results.")
        return
    
    # Create comparison table
    rl_metrics = rl_results['metrics'].iloc[0]
    
    comparison_data = []
    
    # Add MVO
    if 'MVO' in mvo_results:
        comparison_data.append([
            'MVO',
            f"{mvo_results['MVO']['annualized_return']:.2%}",
            f"{mvo_results['MVO']['volatility']:.2%}",
            f"{mvo_results['MVO']['sharpe_ratio']:.2f}",
            f"{mvo_results['MVO']['max_drawdown']:.2%}"
        ])
    
    # Add Deep RL
    comparison_data.append([
        'Deep RL',
        f"{rl_metrics['annualized_return']:.2%}",
        f"{rl_metrics['volatility']:.2%}",
        f"{rl_metrics['sharpe_ratio']:.2f}",
        f"{rl_metrics['max_drawdown']:.2%}"
    ])
    
    # Add Equal Weight
    if 'Equal Weight' in mvo_results:
        comparison_data.append([
            'Equal Weight',
            f"{mvo_results['Equal Weight']['annualized_return']:.2%}",
            f"{mvo_results['Equal Weight']['volatility']:.2%}",
            f"{mvo_results['Equal Weight']['sharpe_ratio']:.2f}",
            f"{mvo_results['Equal Weight']['max_drawdown']:.2%}"
        ])
    
    print("\nStrategy Comparison:")
    print(tabulate(
        comparison_data,
        headers=["Strategy", "Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
        tablefmt="pretty"
    ))
    
    # Show plot if requested
    if show_plot:
        try:
            # We'd need to load the actual cumulative returns data for RL
            # This is a simplified version
            print("Detailed comparison plot not available - requires aligned return data.")
        except Exception as e:
            print(f"Error creating comparison visualization: {e}")

def main():
    parser = argparse.ArgumentParser(description="Portfolio Allocation CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # MVO command
    mvo_parser = subparsers.add_parser("mvo", help="Run Mean-Variance Optimization")
    mvo_parser.add_argument("--risk-aversion", type=float, default=0.1, help="Risk aversion parameter (default: 0.1)")
    mvo_parser.add_argument("--plot", action="store_true", help="Show the performance plot")
    
    # RL results command
    rl_parser = subparsers.add_parser("rl", help="Display Deep RL results")
    rl_parser.add_argument("--plot", action="store_true", help="Show allocation and performance plots")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare different strategies")
    compare_parser.add_argument("--plot", action="store_true", help="Show comparison plot")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "mvo":
        run_mvo(args.risk_aversion, show_plot=args.plot)
    elif args.command == "rl":
        display_rl_results(show_plot=args.plot)
    elif args.command == "compare":
        compare_strategies(show_plot=args.plot)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()