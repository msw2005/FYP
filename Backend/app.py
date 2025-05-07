import os
import pandas as pd
import numpy as np
import sys
import traceback
import platform
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from scipy.optimize import minimize  # Add this import for the minimize function

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
        metrics_file = os.path.join(RESULTS_DIR, "mvo_metrics.csv")
        if os.path.exists(metrics_file):
            # Load the metrics from the CSV file
            metrics_df = pd.read_csv(metrics_file)
            if not metrics_df.empty:
                metrics = metrics_df.iloc[0].to_dict()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'portfolio_value': 10000 * (1 + metrics.get('annualized_return', 0)),
                        'annualized_return': metrics.get('annualized_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'volatility': metrics.get('volatility', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'sortino_ratio': 0,  # This would need to be calculated
                        'turnover': 0.2  # Placeholder value
                    }
                })
        
        # If no metrics file exists, return dummy data
        return jsonify({
            'success': True,
            'data': {
                'portfolio_value': 10500,
                'annualized_return': 0.05,
                'sharpe_ratio': 1.2,
                'volatility': 0.12,
                'max_drawdown': 0.15,
                'sortino_ratio': 1.5,
                'turnover': 0.2
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/portfolio_allocations')
def get_portfolio_allocations():
    try:
        allocations_file = os.path.join(RESULTS_DIR, "mvo_results.csv")
        if os.path.exists(allocations_file):
            # Load the allocations from the results file
            allocations_df = pd.read_csv(allocations_file)
            
            # Format the data for the frontend
            dates = allocations_df['Date'].tolist()
            stock_weights = allocations_df['stock_weight'].tolist()
            bond_weights = allocations_df['bond_weight'].tolist()
            portfolio_values = allocations_df['portfolio_value'].tolist()
            
            return jsonify({
                'success': True,
                'data': {
                    'dates': dates,
                    'stock_weights': stock_weights,
                    'bond_weights': bond_weights,
                    'portfolio_values': portfolio_values
                }
            })
        else:
            # If no results file exists, return dummy data
            dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
            stock_weights = [0.5, 0.52, 0.51]
            bond_weights = [0.5, 0.48, 0.49]
            portfolio_values = [10000, 10100, 10050]
            
            return jsonify({
                'success': True,
                'data': {
                    'dates': dates,
                    'stock_weights': stock_weights,
                    'bond_weights': bond_weights,
                    'portfolio_values': portfolio_values
                }
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
        # Try to load MVO metrics
        mvo_metrics_file = os.path.join(RESULTS_DIR, "mvo_metrics.csv")
        mvo_metrics = None
        if os.path.exists(mvo_metrics_file):
            mvo_df = pd.read_csv(mvo_metrics_file)
            if not mvo_df.empty:
                mvo_metrics = mvo_df.iloc[0].to_dict()
        
        # Try to load Deep RL metrics (if they exist)
        deeprl_metrics_file = os.path.join(RESULTS_DIR, "portfolio_metrics.csv")
        deeprl_metrics = None
        if os.path.exists(deeprl_metrics_file):
            deeprl_df = pd.read_csv(deeprl_metrics_file)
            if not deeprl_df.empty:
                deeprl_metrics = deeprl_df.iloc[0].to_dict()
        
        # Prepare comparison data
        models = []
        returns = []
        sharpe_ratios = []
        volatility = []
        max_drawdown = []
        
        # Add MVO data
        if mvo_metrics:
            models.append("Mean-Variance Optimization")
            returns.append(mvo_metrics.get('annualized_return', 0))
            sharpe_ratios.append(mvo_metrics.get('sharpe_ratio', 0))
            volatility.append(mvo_metrics.get('volatility', 0))
            max_drawdown.append(mvo_metrics.get('max_drawdown', 0))
        else:
            # Add placeholder MVO data
            models.append("Mean-Variance Optimization")
            returns.append(0.06)
            sharpe_ratios.append(0.8)
            volatility.append(0.12)
            max_drawdown.append(0.18)
        
        # Add Deep RL data
        if deeprl_metrics:
            models.append("Deep Reinforcement Learning")
            returns.append(deeprl_metrics.get('annualized_return', 0))
            sharpe_ratios.append(deeprl_metrics.get('sharpe_ratio', 0))
            volatility.append(deeprl_metrics.get('volatility', 0))
            max_drawdown.append(deeprl_metrics.get('max_drawdown', 0))
        else:
            # Add placeholder Deep RL data
            models.append("Deep Reinforcement Learning")
            returns.append(0.08)
            sharpe_ratios.append(1.0)
            volatility.append(0.14)
            max_drawdown.append(0.15)
        
        # Add Equal Weight benchmark
        models.append("Equal Weight (50/50)")
        returns.append(0.05)
        sharpe_ratios.append(0.7)
        volatility.append(0.10)
        max_drawdown.append(0.12)
        
        comparison_data = {
            'models': models,
            'returns': returns,
            'sharpe_ratios': sharpe_ratios,
            'volatility': volatility,
            'max_drawdown': max_drawdown
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
        risk_aversion = float(data.get('risk_aversion', 3.0))
        
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
            "BAMLC0A0CMEY": "Bond", 
            "DGS1MO": "RiskFreeRate"
        }, inplace=True)
        
        # Calculate daily returns
        financial_data["SP500_Return"] = financial_data["SP500"].pct_change()
        financial_data["Bond_Return"] = financial_data["Bond"].pct_change()
        
        # Drop rows with NaN values
        financial_data.dropna(inplace=True)
        
        # Split data into training and testing sets
        train_data = financial_data[financial_data["Date"] < "2020-01-01"]
        test_data = financial_data[financial_data["Date"] >= "2020-01-01"]
        
        # Calculate MVO weights
        train_returns = train_data[["SP500_Return", "Bond_Return"]]
        expected_returns = train_returns.mean().values
        covariance_matrix = train_returns.cov().values
        
        # Run optimization
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
        
        optimal_weights = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion)

        # After calculating optimal_weights:

        # Log the results to show how risk aversion affects the allocation
        print(f"Risk Aversion: {risk_aversion}")
        print(f"Optimal Stock Weight: {optimal_weights[0]}")
        print(f"Optimal Bond Weight: {optimal_weights[1]}")
        
        # Backtest the optimal weights on test data
        portfolio_values = [10000]  # Start with $10,000
        stock_weights = []
        bond_weights = []
        dates = []
        
        for i in range(len(test_data) - 1):
            current_date = test_data.iloc[i]['Date']
            next_date = test_data.iloc[i + 1]['Date']
            next_stock_return = test_data.iloc[i + 1]['SP500_Return']
            next_bond_return = test_data.iloc[i + 1]['Bond_Return']
            
            # Calculate portfolio return for this period
            portfolio_return = optimal_weights[0] * next_stock_return + optimal_weights[1] * next_bond_return
            
            # Update portfolio value
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
            
            # Store allocation and date
            stock_weights.append(optimal_weights[0])
            bond_weights.append(optimal_weights[1])
            dates.append(next_date.strftime('%Y-%m-%d'))
        
        # Calculate performance metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = abs(min(drawdown))
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'Date': dates,
            'stock_weight': stock_weights,
            'bond_weight': bond_weights,
            'portfolio_value': portfolio_values[1:]
        })
        
        results_file = os.path.join(RESULTS_DIR, 'mvo_results.csv')
        results_df.to_csv(results_file, index=False)
        
        # Save metrics
        metrics = {
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'optimal_weights': [float(w) for w in optimal_weights]
        }
        
        metrics_file = os.path.join(RESULTS_DIR, 'mvo_metrics.csv')
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        
        return jsonify({
            'success': True,
            'message': 'MVO model executed successfully',
            'metrics': metrics,
            'results_file': results_file
        })
        
    except Exception as e:
        import traceback
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

# In app.py - add a route to check job status
@app.route('/api/job_status/<job_id>', methods=['GET'])
def check_job_status(job_id):
    try:
        # This is a simplified version - in a real app you'd track job status in a database
        # For now, we'll just check if result files exist
        metrics_file = os.path.join(RESULTS_DIR, f"{job_id}_metrics.csv")
        if os.path.exists(metrics_file):
            return jsonify({
                'success': True,
                'status': 'completed',
                'message': 'Model finished processing'
            })
        else:
            return jsonify({
                'success': True,
                'status': 'running',
                'message': 'Model is still processing'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# New endpoint to run Deep RL model

@app.route('/api/run_deep_rl', methods=['POST'])
def run_deep_rl():
    try:
        # Get parameters from request
        data = request.get_json()
        risk_aversion = float(data.get('risk_aversion', 0.1))
        
        # For this to work properly in production, you would need to:
        # 1. Extract the relevant code from models.ipynb to a .py file
        # 2. Import and call that code here
        
        # For now, we'll run a subprocess that calls the existing Python script
        import subprocess
        import uuid
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Create a script that runs the Deep RL model
        script_path = os.path.join(MODELS_DIR, f"run_deeprl_{job_id}.py")
        with open(script_path, 'w') as f:
            f.write(f"""
import os
import sys
import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
import matplotlib.pyplot as plt

# Set paths
BASE_DIR = "{BASE_DIR}"
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
MODEL_DIR = os.path.join(BASE_DIR, "Models", "DeepRL")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Calculate maximum drawdown
def calculate_max_drawdown(portfolio_values):
    portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) <= 1:
        return 0.0
    
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / np.maximum(cumulative_max, 1e-10)
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    return abs(max_drawdown)

# Custom Environment for Portfolio Allocation
class PortfolioAllocationEnv(gym.Env):
    def __init__(self, data, risk_aversion=3.0, window_size=30):
        super(PortfolioAllocationEnv, self).__init__()
        
        # Make a copy of the data to avoid modifying the original
        self.data = data.reset_index(drop=True).copy()
        self.risk_aversion = risk_aversion
        self.window_size = window_size
        
        # Ensure data has enough rows
        if len(self.data) <= window_size + 10:
            raise ValueError(f"Data has {{len(self.data)}} rows, need more than {{window_size + 10}}")
        
        self.current_step = window_size
        self.max_steps = len(self.data) - 1
        
        # Define action space: continuous allocation between assets
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        # Define observation space with market features
        n_features = 7  # Using 7 features for state representation
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(n_features,), dtype=np.float32
        )
        
        # Track portfolio performance
        self.portfolio_value = 10000  # Initial portfolio value
        self.portfolio_history = [self.portfolio_value]
        self.weights_history = []
        self.returns_history = []
    
    def reset(self):
        self.current_step = self.window_size
        self.portfolio_value = 10000
        self.portfolio_history = [self.portfolio_value]
        self.weights_history = []
        self.returns_history = []
        return self._get_observation()
    
    def _get_observation(self):
        try:
            obs = np.array([
                float(self.data.loc[self.current_step, "SP500_Return"]),
                float(self.data.loc[self.current_step, "Bond_Return"]),
                float(self.data.loc[self.current_step, "SP500_Volatility"]),
                float(self.data.loc[self.current_step, "Bond_Volatility"]),
                float(self.data.loc[self.current_step, "SP500_RelToMA"]),
                float(self.data.loc[self.current_step, "Bond_RelToMA"]),
                float(self.data.loc[self.current_step, "RiskFreeRate"]) / 100.0
            ], dtype=np.float32)
            
            # Check for NaN or Inf values
            if np.isnan(obs).any() or np.isinf(obs).any():
                print(f"Warning: NaN or Inf in observation at step {{self.current_step}}, replacing with zeros")
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip values to ensure they are within reasonable bounds
            obs = np.clip(obs, -5.0, 5.0)
            
            return obs
        except Exception as e:
            print(f"Error getting observation at step {{self.current_step}}: {{e}}")
            # Return a safe default observation
            return np.zeros(7, dtype=np.float32)
    
    def step(self, action):
        # Extract the weight for stocks from action
        stock_weight = float(np.clip(action[0], 0.0, 1.0))
        bond_weight = 1.0 - stock_weight
        
        # Store current weights
        self.weights_history.append([stock_weight, bond_weight])
        
        # Calculate returns for next step
        if self.current_step < self.max_steps:
            try:
                next_stock_return = float(self.data.loc[self.current_step + 1, "SP500_Return"])
                next_bond_return = float(self.data.loc[self.current_step + 1, "Bond_Return"])
                risk_free_rate = float(self.data.loc[self.current_step, "RiskFreeRate"]) / 100.0 / 252.0
                
                # Safety checks - replace NaN values
                if np.isnan(next_stock_return) or np.isinf(next_stock_return):
                    next_stock_return = 0.0
                if np.isnan(next_bond_return) or np.isinf(next_bond_return):
                    next_bond_return = 0.0
                if np.isnan(risk_free_rate) or np.isinf(risk_free_rate):
                    risk_free_rate = 0.0
                
                # Clip returns to prevent extreme values
                next_stock_return = np.clip(next_stock_return, -0.1, 0.1)
                next_bond_return = np.clip(next_bond_return, -0.1, 0.1)
                
                # Calculate portfolio return
                portfolio_return = stock_weight * next_stock_return + bond_weight * next_bond_return
                
                self.returns_history.append(portfolio_return)
                
                # Update portfolio value
                self.portfolio_value *= (1.0 + portfolio_return)
                self.portfolio_history.append(self.portfolio_value)
                
                # Calculate reward
                excess_return = portfolio_return - risk_free_rate
                
                if len(self.returns_history) >= 20:
                    recent_returns = np.array(self.returns_history[-20:])
                    volatility = np.std(recent_returns)
                    
                    recent_portfolio = np.array(self.portfolio_history[-20:])
                    if len(recent_portfolio) > 1:
                        peak = np.maximum.accumulate(recent_portfolio)
                        drawdown = (recent_portfolio - peak) / np.maximum(peak, 1e-10)
                        max_drawdown = abs(np.min(drawdown))
                    else:
                        max_drawdown = 0.0
                    
                    # Adjust reward based on risk aversion
                    reward = (
                        excess_return - 
                        self.risk_aversion * volatility**2 - 
                        self.risk_aversion * max_drawdown * 0.5
                    )
                else:
                    reward = excess_return
                
                # Clip reward to reasonable range
                reward = np.clip(reward, -1.0, 1.0)
                
                # Move to the next step
                self.current_step += 1
                done = self.current_step >= self.max_steps
                
                return self._get_observation(), float(reward), done, {{
                    "portfolio_value": float(self.portfolio_value),
                    "weights": [float(stock_weight), float(bond_weight)],
                    "returns": float(portfolio_return)
                }}
            
            except Exception as e:
                print(f"Error in environment step: {{e}}")
                self.current_step += 1
                return self._get_observation(), 0.0, True, {{
                    "portfolio_value": float(self.portfolio_value),
                    "weights": [float(stock_weight), float(bond_weight)],
                    "returns": 0.0
                }}
        
        else:
            # If we've reached the end of the data
            return self._get_observation(), 0.0, True, {{
                "portfolio_value": float(self.portfolio_value),
                "weights": [float(stock_weight), float(bond_weight)],
                "returns": 0.0
            }}
    
    def render(self, mode='human'):
        print(f"Step: {{self.current_step}}, Portfolio Value: {{self.portfolio_value:.2f}}")

# Load and preprocess data
def load_and_preprocess_data(split_date="2020-01-01", risk_aversion_override=None):
    # Load datasets
    sp500_file = os.path.join(DATA_DIR, "processed_sp500_data.csv")
    bond_file = os.path.join(DATA_DIR, "processed_bond_data.csv")
    risk_free_file = os.path.join(DATA_DIR, "processed_risk_free_rate_data.csv")
    
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
    financial_data.rename(columns={{
        "SP500_Close": "SP500", 
        "BAMLC0A0CMEY": "Bond", 
        "DGS1MO": "RiskFreeRate"
    }}, inplace=True)
    
    # Handle missing values
    financial_data = financial_data.ffill().bfill()
    
    # Calculate daily returns
    financial_data["SP500_Return"] = financial_data["SP500"].pct_change()
    financial_data["Bond_Return"] = financial_data["Bond"].pct_change()
    
    # Handle extreme returns by clipping
    financial_data["SP500_Return"] = financial_data["SP500_Return"].clip(-0.1, 0.1)
    financial_data["Bond_Return"] = financial_data["Bond_Return"].clip(-0.1, 0.1)
    
    # Calculate technical indicators
    # 1. Moving averages
    financial_data["SP500_MA30"] = financial_data["SP500"].rolling(window=30).mean()
    financial_data["Bond_MA30"] = financial_data["Bond"].rolling(window=30).mean()
    
    # 2. Volatility (standard deviation over rolling window)
    financial_data["SP500_Volatility"] = financial_data["SP500_Return"].rolling(window=20).std()
    financial_data["Bond_Volatility"] = financial_data["Bond_Return"].rolling(window=20).std()
    
    # 3. Price relative to moving average (normalized)
    financial_data["SP500_RelToMA"] = (financial_data["SP500"] / financial_data["SP500_MA30"]) - 1
    financial_data["Bond_RelToMA"] = (financial_data["Bond"] / financial_data["Bond_MA30"]) - 1
    
    # Drop rows with NaN values
    financial_data.dropna(inplace=True)
    
    # Replace infinite values with something reasonable
    for col in financial_data.columns:
        if financial_data[col].dtype != 'object' and financial_data[col].dtype != 'datetime64[ns]':
            # Replace infinities with large but finite values
            financial_data[col] = financial_data[col].replace([np.inf, -np.inf], [1.0, -1.0])
    
    # Check for any remaining NaN or infinite values
    feature_columns = [
        "SP500_Return", "Bond_Return", "SP500_Volatility", "Bond_Volatility",
        "SP500_RelToMA", "Bond_RelToMA", "RiskFreeRate"
    ]
    
    for col in feature_columns:
        if financial_data[col].isna().any() or np.isinf(financial_data[col]).any():
            print(f"Warning: Column {{col}} contains NaN or Inf values. Replacing with zeros.")
            financial_data[col] = financial_data[col].replace([np.inf, -np.inf], [1.0, -1.0])
            financial_data[col] = financial_data[col].fillna(0)
    
    # Normalize features for RL
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    financial_data[feature_columns] = scaler.fit_transform(financial_data[feature_columns])
    
    # Additional safety check - cap normalized values to prevent extreme outliers
    for col in feature_columns:
        financial_data[col] = financial_data[col].clip(-3, 3)
    
    # Split data into train and test sets
    train_data = financial_data[financial_data["Date"] < split_date].copy()
    test_data = financial_data[financial_data["Date"] >= split_date].copy()
    
    # Set risk aversion based on parameter or default
    if risk_aversion_override is not None:
        risk_aversion = risk_aversion_override
    else:
        # Load risk aversion from inverse optimization results if available
        try:
            inverse_opt_results = pd.read_csv(os.path.join(RESULTS_DIR, "inverse_optimization_results.csv"))
            if not inverse_opt_results.empty and "Risk_Aversion" in inverse_opt_results.columns:
                best_idx = inverse_opt_results["Error"].idxmin()
                risk_aversion = inverse_opt_results.iloc[best_idx]["Risk_Aversion"]
            else:
                risk_aversion = {risk_aversion}  # Use the passed-in parameter
        except:
            risk_aversion = {risk_aversion}  # Use the passed-in parameter
    
    return train_data, test_data, risk_aversion, scaler

def train_portfolio_rl(train_data, risk_aversion, total_timesteps=10000, save_path=None):
    # Create environment
    env = DummyVecEnv([lambda: PortfolioAllocationEnv(train_data, risk_aversion=risk_aversion)])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,
        gamma=0.95,
        n_steps=1024,
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 32], vf=[64, 32])],
            activation_fn=torch.nn.ReLU
        )
    )
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model if path is provided
    if save_path:
        model.save(save_path)
        print(f"Model saved to {{save_path}}")
    
    return model

def evaluate_model(model, test_data, risk_aversion):
    # Create test environment
    env = PortfolioAllocationEnv(test_data, risk_aversion=risk_aversion)
    
    # Reset environment
    obs = env.reset()
    
    # Initialize tracking variables
    portfolio_values = [env.portfolio_value]
    stock_weights = []
    bond_weights = []
    returns = []
    dates = test_data["Date"].values[env.window_size:] if "Date" in test_data.columns else None
    
    # Run evaluation until done
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        stock_weights.append(info["weights"][0])
        bond_weights.append(info["weights"][1])
        returns.append(info["returns"])
    
    # Calculate performance metrics
    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
    annualized_return = ((1 + cumulative_returns[-1]) ** (252 / len(returns))) - 1
    volatility = np.std(returns) * np.sqrt(252)
    
    # Calculate risk-adjusted metrics
    avg_rf_rate = test_data["RiskFreeRate"].mean() / 100
    sharpe_ratio = (annualized_return - avg_rf_rate) / volatility if volatility > 0 else 0
    
    # Calculate sortino ratio (penalizes only downside volatility)
    downside_returns = np.array([min(r, 0) for r in returns])
    downside_volatility = np.std(downside_returns) * np.sqrt(252)
    sortino_ratio = (annualized_return - avg_rf_rate) / downside_volatility if downside_volatility > 0 else 0
    
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Calculate turnover (trading activity)
    weight_changes = np.abs(np.diff(np.array(stock_weights)))
    turnover = np.sum(weight_changes)
    
    return {{
        "portfolio_values": portfolio_values,
        "stock_weights": stock_weights, 
        "bond_weights": bond_weights,
        "returns": returns,
        "dates": dates,
        "metrics": {{
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "turnover": float(turnover)
        }}
    }}

# Main function
def main():
    # Load and preprocess data with specified risk aversion
    train_data, test_data, risk_aversion, scaler = load_and_preprocess_data(risk_aversion_override={risk_aversion})
    print(f"Using risk aversion parameter: {{risk_aversion}}")
    
    # Define file paths
    model_path = os.path.join(MODEL_DIR, "ppo_portfolio_model.zip")
    
    # Train model (reduced timesteps to fit within timeout)
    model = train_portfolio_rl(
        train_data, 
        risk_aversion=risk_aversion,
        total_timesteps=10000,  # Reduced for quick testing
        save_path=model_path
    )
    
    # Define benchmark weights for comparison
    benchmark_weights = {{
        "Equal Weight": [0.5, 0.5],  # 50% stocks, 50% bonds
        "MVO": [0.6, 0.4],  # From traditional MVO optimization
        "Observed Weights": [0.68, 0.32]  # From fund weights observed in the data
    }}
    
    # Evaluate model
    evaluation_results = evaluate_model(
        model, 
        test_data, 
        risk_aversion=risk_aversion
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([evaluation_results["metrics"]])
    metrics_df["risk_aversion"] = risk_aversion
    metrics_file = os.path.join(RESULTS_DIR, "portfolio_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save allocation weights to CSV for later analysis
    min_length = min(
        len(evaluation_results["stock_weights"]),
        len(evaluation_results["bond_weights"]),
        len(evaluation_results["portfolio_values"]) - 1  # Skip initial value
    )
    
    # Create dates array with matching length
    if evaluation_results["dates"] is not None and len(evaluation_results["dates"]) >= min_length:
        dates = evaluation_results["dates"][:min_length]
    else:
        dates = range(min_length)
    
    weights_df = pd.DataFrame({{
        "Date": dates,
        "Stock Weight": evaluation_results["stock_weights"][:min_length],
        "Bond Weight": evaluation_results["bond_weights"][:min_length],
        "Portfolio Value": evaluation_results["portfolio_values"][1:min_length+1]  # Skip initial value
    }})
    weights_file = os.path.join(RESULTS_DIR, "portfolio_allocations.csv")
    weights_df.to_csv(weights_file, index=False)
    print(f"Evaluation complete! Results saved to {{RESULTS_DIR}}")

if __name__ == "__main__":
    main()
            """)
        
        # Run the script as a subprocess with a longer timeout to be safer
        result = subprocess.run(
            ["python", script_path], 
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        # Clean up the temporary script
        try:
            os.remove(script_path)
        except:
            pass  # Ignore cleanup errors
        
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
                'message': 'Deep RL model executed successfully',
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