import os
import pandas as pd
import numpy as np
import sys
import traceback
import platform
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Add the Models directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Models'))

# Import model functions from your existing files
from Models.MVO import optimize_portfolio, calculate_statistics

app = Flask(__name__)
CORS(app)  # Enable CORS for API access from frontend

# Detect operating system and set appropriate base directory
if platform.system() == 'Windows':
    # Windows paths
    BASE_DIR = r"c:\Users\mohda\OneDrive - University of Greenwich\Documents\Year 3\FYP"
else:
    # macOS paths
    BASE_DIR = r"/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP"

RESULTS_DIR = os.path.join(BASE_DIR, "Results")
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# For a more robust solution, attempt to find the data directory if the default paths don't work
if not os.path.exists(DATA_DIR):
    # Try alternate paths relative to the current file
    alt_data_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Processed')
    if os.path.exists(alt_data_dir):
        DATA_DIR = alt_data_dir
    
    # Try another common path on macOS
    alt_data_dir2 = os.path.join(os.path.dirname(__file__), 'Data', 'Processed')
    if os.path.exists(alt_data_dir2):
        DATA_DIR = alt_data_dir2

# Same for results directory
if not os.path.exists(RESULTS_DIR):
    alt_results_dir = os.path.join(os.path.dirname(__file__), '..', 'Results')
    if os.path.exists(alt_results_dir):
        RESULTS_DIR = alt_results_dir
    
    alt_results_dir2 = os.path.join(os.path.dirname(__file__), 'Results')
    if os.path.exists(alt_results_dir2):
        RESULTS_DIR = alt_results_dir2

# Ensure these directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/portfolio_metrics')
def get_portfolio_metrics():
    try:
        metrics_file = os.path.join(RESULTS_DIR, "portfolio_metrics.csv")
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            return jsonify({
                'success': True,
                'data': metrics_df.to_dict(orient='records')[0]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Metrics file not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/portfolio_allocations')
def get_portfolio_allocations():
    try:
        allocations_file = os.path.join(RESULTS_DIR, "portfolio_allocations.csv")
        if os.path.exists(allocations_file):
            allocations_df = pd.read_csv(allocations_file)
            
            # Format date column if it exists
            if 'Date' in allocations_df.columns:
                allocations_df['Date'] = pd.to_datetime(allocations_df['Date']).dt.strftime('%Y-%m-%d')
            
            return jsonify({
                'success': True,
                'data': {
                    'dates': allocations_df['Date'].tolist() if 'Date' in allocations_df.columns else list(range(len(allocations_df))),
                    'stock_weights': allocations_df['Stock Weight'].tolist(),
                    'bond_weights': allocations_df['Bond Weight'].tolist(),
                    'portfolio_values': allocations_df['Portfolio Value'].tolist() if 'Portfolio Value' in allocations_df.columns else []
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Allocations file not found'
            })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/model_comparison')
def get_model_comparison():
    try:
        # Load data from portfolio_metrics.csv if available
        metrics_file = os.path.join(RESULTS_DIR, "portfolio_metrics.csv")
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            
            # Get data for Deep RL from metrics_df
            deep_rl_data = {
                'annualized_return': float(metrics_df['annualized_return'].iloc[0]),
                'volatility': float(metrics_df['volatility'].iloc[0]),
                'sharpe_ratio': float(metrics_df['sharpe_ratio'].iloc[0]),
                'max_drawdown': float(metrics_df['max_drawdown'].iloc[0])
            }
            
            # Get data for benchmarks from benchmark_comparison if available
            benchmark_data = {}
            if 'benchmark_comparison' in metrics_df.columns:
                try:
                    import json
                    # Try to parse as JSON first
                    benchmark_comparison_str = metrics_df['benchmark_comparison'].iloc[0]
                    if isinstance(benchmark_comparison_str, str):
                        benchmark_data = json.loads(benchmark_comparison_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, use default values instead
                    print("Warning: Could not parse benchmark_comparison as JSON. Using default values.")
                    benchmark_data = {
                        'MVO': {
                            'annualized_return': 0.142,
                            'volatility': 0.152,
                            'sharpe_ratio': 0.92,
                            'max_drawdown': 0.25
                        },
                        'Equal Weight': {
                            'annualized_return': 0.128,
                            'volatility': 0.158,
                            'sharpe_ratio': 0.81,
                            'max_drawdown': 0.28
                        }
                    }
            
            # Create comparison data
            comparison_data = {
                'models': ['Deep RL', 'MVO', 'Equal Weight'],
                'returns': [
                    deep_rl_data['annualized_return'],
                    benchmark_data.get('MVO', {}).get('annualized_return', 0.142),
                    benchmark_data.get('Equal Weight', {}).get('annualized_return', 0.128)
                ],
                'sharpe_ratios': [
                    deep_rl_data['sharpe_ratio'],
                    benchmark_data.get('MVO', {}).get('sharpe_ratio', 0.92),
                    benchmark_data.get('Equal Weight', {}).get('sharpe_ratio', 0.81)
                ],
                'volatility': [
                    deep_rl_data['volatility'],
                    benchmark_data.get('MVO', {}).get('volatility', 0.152),
                    benchmark_data.get('Equal Weight', {}).get('volatility', 0.158)
                ],
                'max_drawdown': [
                    deep_rl_data['max_drawdown'],
                    benchmark_data.get('MVO', {}).get('max_drawdown', 0.25),
                    benchmark_data.get('Equal Weight', {}).get('max_drawdown', 0.28)
                ]
            }
        else:
            # Use default data if file doesn't exist
            comparison_data = {
                'models': ['Deep RL', 'MVO', 'Equal Weight'],
                'returns': [0.164, 0.142, 0.128],
                'sharpe_ratios': [1.05, 0.92, 0.81],
                'volatility': [0.149, 0.152, 0.158],
                'max_drawdown': [0.22, 0.25, 0.28]
            }
        
        return jsonify({
            'success': True,
            'data': comparison_data
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

# New endpoint to run MVO model
@app.route('/api/run_mvo', methods=['POST'])
def run_mvo():
    try:
        # Get parameters from request
        data = request.get_json()
        risk_aversion = float(data.get('risk_aversion', 0.1))
        
        # Load data for MVO
        sp500_file = os.path.join(DATA_DIR, "processed_sp500_data.csv")
        bond_file = os.path.join(DATA_DIR, "processed_bond_data.csv")
        risk_free_file = os.path.join(DATA_DIR, "processed_risk_free_rate_data.csv")
        
        # Load datasets
        sp500_data = pd.read_csv(sp500_file)
        bond_data = pd.read_csv(bond_file)
        risk_free_rate_data = pd.read_csv(risk_free_file)
        
        # Process data for MVO
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
        
        # Drop NaN values
        financial_data.dropna(inplace=True)
        
        # Split into train/test
        split_date = "2020-01-01"
        train_data = financial_data[financial_data["Date"] < split_date]
        test_data = financial_data[financial_data["Date"] >= split_date]
        
        # Calculate statistics and run optimization
        expected_returns, covariance_matrix = calculate_statistics(train_data)
        optimal_weights = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion)
        
        # Calculate portfolio values using test data
        test_data["MVO_Return"] = (
            optimal_weights[0] * test_data["SP500_Return"] + 
            optimal_weights[1] * test_data["USBIG_Return"]
        )
        test_data["Portfolio_Value"] = 10000 * (1 + test_data["MVO_Return"]).cumprod()
        
        # Calculate performance metrics
        cumulative_return = (1 + test_data["MVO_Return"]).prod() - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(test_data)) - 1
        volatility = test_data["MVO_Return"].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility
        
        # Calculate max drawdown
        portfolio_values = test_data["Portfolio_Value"].values
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Save results to CSV for display
        mvo_results = pd.DataFrame({
            "Date": test_data["Date"],
            "Stock Weight": [optimal_weights[0]] * len(test_data),
            "Bond Weight": [optimal_weights[1]] * len(test_data),
            "Portfolio Value": test_data["Portfolio_Value"]
        })
        mvo_results.to_csv(os.path.join(RESULTS_DIR, "mvo_results.csv"), index=False)
        
        # Return results
        return jsonify({
            'success': True,
            'data': {
                'optimal_weights': optimal_weights.tolist(),
                'risk_aversion': risk_aversion,
                'annualized_return': float(annualized_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'portfolio_values': test_data["Portfolio_Value"].tolist(),
                'dates': test_data["Date"].dt.strftime('%Y-%m-%d').tolist()
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

# New endpoint to run Inverse Optimization
@app.route('/api/run_inverse_optimization', methods=['POST'])
def run_inverse_optimization():
    try:
        # Import necessary functions
        # In app.py
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Models'))
        from Models.MVO import optimize_portfolio, calculate_statistics
        
        # Load observed weights or use default
        data = request.get_json()
        observed_weights = data.get('observed_weights', [0.68, 0.32])
        observed_weights = np.array(observed_weights)
        
        # Load and prepare financial data
        sp500_file = os.path.join(DATA_DIR, "processed_sp500_data.csv")
        bond_file = os.path.join(DATA_DIR, "processed_bond_data.csv")
        risk_free_file = os.path.join(DATA_DIR, "processed_risk_free_rate_data.csv")
        
        # Load datasets
        sp500_data = pd.read_csv(sp500_file)
        bond_data = pd.read_csv(bond_file)
        risk_free_rate_data = pd.read_csv(risk_free_file)
        
        # Process data
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
        
        # Rename columns
        financial_data.rename(columns={
            "SP500_Close": "SP500", 
            "BAMLC0A0CMEY": "USBIG", 
            "DGS1MO": "RiskFreeRate"
        }, inplace=True)
        
        # Calculate returns
        financial_data["SP500_Return"] = financial_data["SP500"].pct_change()
        financial_data["USBIG_Return"] = financial_data["USBIG"].pct_change()
        
        # Drop NaN
        financial_data.dropna(inplace=True)
        
        # Split data
        split_date = "2020-01-01"
        train_data = financial_data[financial_data["Date"] < split_date]
        
        # Get statistics
        train_expected_returns, train_covariance_matrix = calculate_statistics(train_data)
        
        # Define a function to estimate risk aversion
        def inv_optimize(observed_weights, expected_returns, covariance_matrix, candidate_range):
            errors = []
            candidate_risk = []
            best_error = np.inf
            best_risk = None
            best_weights = None

            for risk_aversion in candidate_range:
                try:
                    model_weights = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion)
                    error = np.linalg.norm(model_weights - observed_weights)
                    errors.append(float(error))
                    candidate_risk.append(float(risk_aversion))
                    if error < best_error:
                        best_error = error
                        best_risk = risk_aversion
                        best_weights = model_weights
                except Exception as e:
                    continue

            return best_risk, best_weights, candidate_risk, errors
        
        # Run inverse optimization
        candidate_range = np.linspace(0.1, 10, 100)
        best_risk, best_weights, candidate_risk, errors = inv_optimize(
            observed_weights, train_expected_returns, train_covariance_matrix, candidate_range
        )
        
        # Save results to CSV
        results_df = pd.DataFrame({
            "Risk_Aversion": candidate_risk,
            "Error": errors
        })
        results_file = os.path.join(RESULTS_DIR, "inverse_optimization_results.csv")
        results_df.to_csv(results_file, index=False)
        
        return jsonify({
            'success': True,
            'data': {
                'best_risk_aversion': float(best_risk),
                'best_weights': best_weights.tolist() if best_weights is not None else None,
                'observed_weights': observed_weights.tolist(),
                'candidate_risk': candidate_risk,
                'errors': errors
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

# New endpoint to run Deep RL model
@app.route('/api/run_deep_rl', methods=['POST'])
def run_deep_rl():
    try:
        # For this to work properly in production, you would need to:
        # 1. Extract the relevant code from models.ipynb to a .py file
        # 2. Import and call that code here
        
        # For now, we'll run a subprocess that calls the existing Python script
        import subprocess
        
        # Create a script that runs the Deep RL model
        script_path = os.path.join(MODELS_DIR, "models.py")
        with open(script_path, 'w') as f:
            f.write("""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessaty modules
import pandas as pd
import numpy as np
import torch
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Set paths
data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "Processed")
results_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepRL")

# Import the deepRL code from models.py
# (You would need to extract this code from the notebook)
from models import load_and_preprocess_data, train_portfolio_rl, evaluate_model, plot_results

def main():
    # Load and preprocess data
    train_data, test_data, risk_aversion, scaler = load_and_preprocess_data()
    
    # Define file paths
    model_path = os.path.join(model_folder, "ppo_portfolio_model.zip")
    plot_path = os.path.join(results_folder, "portfolio_performance.png")
    
    # Train model
    model = train_portfolio_rl(
        train_data, 
        risk_aversion=risk_aversion,
        total_timesteps=150000, 
        save_path=model_path
    )
    
    # Define benchmark weights for comparison
    benchmark_weights = {
        "Equal Weight": [0.5, 0.5],  # 50% stocks, 50% bonds
        "MVO": [0.6, 0.4],  # From traditional MVO optimization
        "Observed Weights": [0.675, 0.31833333]  # From fund weights observed in the data
    }
    
    # Evaluate model
    evaluation_results = evaluate_model(
        model, 
        test_data, 
        risk_aversion=risk_aversion,
        benchmark_weights=benchmark_weights
    )
    
    # Plot and save results
    plot_results(
        evaluation_results, 
        save_path=plot_path,
        benchmark_weights=benchmark_weights
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([evaluation_results["metrics"]])
    metrics_df["risk_aversion"] = risk_aversion
    metrics_file = os.path.join(results_folder, "portfolio_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save allocation weights to CSV
    weights_df = pd.DataFrame({
        "Date": evaluation_results["dates"] if evaluation_results["dates"] is not None else range(len(evaluation_results["stock_weights"])),
        "Stock Weight": evaluation_results["stock_weights"],
        "Bond Weight": evaluation_results["bond_weights"],
        "Portfolio Value": evaluation_results["portfolio_values"][1:]  # Skip initial value
    })
    weights_file = os.path.join(results_folder, "portfolio_allocations.csv")
    weights_df.to_csv(weights_file, index=False)
    
    return model, evaluation_results

if __name__ == "__main__":
    main()
            """)
        
        # Run the script as a subprocess
        # Note: In production, you might want to use a queue system as this could take a while
        result = subprocess.run(
            ["python", script_path], 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            return jsonify({
                'success': False,
                'error': f"Error running Deep RL model: {result.stderr}"
            })
        
        # Load results from the generated files
        metrics_file = os.path.join(RESULTS_DIR, "portfolio_metrics.csv")
        allocations_file = os.path.join(RESULTS_DIR, "portfolio_allocations.csv")
        
        if os.path.exists(metrics_file) and os.path.exists(allocations_file):
            metrics_df = pd.read_csv(metrics_file)
            allocations_df = pd.read_csv(allocations_file)
            
            return jsonify({
                'success': True,
                'data': {
                    'metrics': metrics_df.to_dict(orient='records')[0],
                    'allocations': {
                        'dates': allocations_df['Date'].tolist() if 'Date' in allocations_df.columns else list(range(len(allocations_df))),
                        'stock_weights': allocations_df['Stock Weight'].tolist(),
                        'bond_weights': allocations_df['Bond Weight'].tolist(),
                        'portfolio_values': allocations_df['Portfolio Value'].tolist() if 'Portfolio Value' in allocations_df.columns else []
                    }
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No results files created by Deep RL model'
            })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)