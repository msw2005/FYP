import os
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces

def preprocess_datasets(data_folder):
    """
    Preprocess datasets to extract relevant data for Deep RL training.

    Parameters:
        data_folder (str): Path to the folder containing the datasets.

    Returns:
        pd.DataFrame: Preprocessed financial data with macroeconomic factors and technical indicators.
    """
    files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]
    financial_data = pd.DataFrame()
    macroeconomic_data = pd.DataFrame()

    for file in files:
        file_path = os.path.join(data_folder, file)
        excel_data = pd.ExcelFile(file_path, engine='openpyxl')

        for sheet_name in excel_data.sheet_names:
            if sheet_name.lower() in ['explanations and faq', 'summary for ppt']:
                continue

            df = excel_data.parse(sheet_name)
            if df.empty:
                continue

            if 'Date' in df.columns or 'Year' in df.columns:
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index, errors='coerce')
                elif 'Year' in df.columns:
                    df.set_index('Year', inplace=True)

                df = df[~df.index.duplicated(keep='first')]
                numeric_data = df.select_dtypes(include=[np.number])

                # Strip whitespace from column names
                numeric_data.columns = numeric_data.columns.str.strip()

                # Separate equity data and macroeconomic data
                equity_columns = ['Price', 'Return', 'S&P 500 (includes dividends)', 'Price per oz', 'S&P 500 (Real)', 'T. Bills']
                macro_columns = ['CPI', 'Unemployment', '1-Month Treasury']

                equity_data = numeric_data[[col for col in numeric_data.columns if col in equity_columns]]
                macro_data = numeric_data[[col for col in numeric_data.columns if col in macro_columns]]

                financial_data = pd.concat([financial_data, equity_data], axis=1)
                macroeconomic_data = pd.concat([macroeconomic_data, macro_data], axis=1)

    if not financial_data.empty:
        # Handle missing data (e.g., forward fill)
        financial_data = financial_data.ffill().bfill()
        macroeconomic_data = macroeconomic_data.ffill().bfill()

        # Ensure the index is sorted
        financial_data = financial_data.sort_index()
        macroeconomic_data = macroeconomic_data.sort_index()

        # Combine equity and macroeconomic data
        combined_data = pd.concat([financial_data, macroeconomic_data], axis=1)

        # Add technical indicators if 'Price' column exists
        if 'Price' in combined_data.columns:
            combined_data['SMA'] = combined_data['Price'].rolling(window=10).mean()  # Simple Moving Average
            combined_data['RSI'] = calculate_rsi(combined_data['Price'])  # Relative Strength Index
        else:
            print("Warning: 'Price' column not found. Technical indicators will not be added.")

        return combined_data

    raise ValueError("No valid financial data found in the datasets.")

def calculate_rsi(prices, window=14):
    """
    Calculate the Relative Strength Index (RSI).

    Parameters:
        prices (pd.Series): Series of prices.
        window (int): Lookback window for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class CustomEnv(gym.Env):
    """
    Custom Environment for Deep RL based on the dataset.
    """
    def __init__(self, data, render_mode=None):
        super(CustomEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.render_mode = render_mode

        # Define action and observation space
        # Actions: Discrete actions (e.g., Buy, Sell, Hold)
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell

        # Observations: State space (e.g., price, returns, macroeconomic factors, technical indicators)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
        )

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        """
        Take an action and return the next state, reward, done, and info.
        """
        reward = 0
        done = False

        # Get the current return from the dataset
        current_return = self.data.iloc[self.current_step]['Return']

        # Reward logic (example: profit/loss based on action)
        if action == 1:  # Buy
            reward = current_return - 0.01 * abs(current_return)  # Penalize large losses
        elif action == 2:  # Sell
            reward = -current_return - 0.01 * abs(current_return)

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        # Get the next state
        next_state = self.data.iloc[self.current_step].values

        # Include 'Return' in the info dictionary
        info = {"Return": current_return}

        return next_state, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment (optional).
        """
        print(f"Step: {self.current_step}")

# Path to the data folder
data_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data"

# Preprocess the dataset
data = preprocess_datasets(data_folder)

# Normalize the data
data.fillna(0, inplace=True)  # Handle missing values

# Create the custom environment
env = DummyVecEnv([lambda: CustomEnv(data)])

# Define the RL model (PPO)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, gamma=0.99, n_steps=2048)

# Train the model
timesteps = 200000  # Increased training timesteps
model.learn(total_timesteps=timesteps)

# Save the trained model
model_dir = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Models/Deep RL1"
model_path = os.path.join(model_dir, "ppo_model.zip")
os.makedirs(model_dir, exist_ok=True)
model.save(model_path)

# Evaluate the model
def evaluate_model(env, model, risk_free_rate=0.01, render=False):
    """
    Evaluate the performance of the trained RL model.

    Parameters:
        env (gym.Env): The environment used for evaluation.
        model (stable_baselines3.PPO): The trained RL model.
        risk_free_rate (float): The risk-free rate for Sharpe Ratio calculation.
        render (bool): Whether to render the environment during evaluation.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    obs = env.reset()
    total_rewards = []
    portfolio_returns = []
    actions = []
    done = False

    while not done:
        # Predict the action using the trained model
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Collect metrics
        total_rewards.append(reward)
        actions.append(action)
        if 'Return' in info:
            portfolio_returns.append(info['Return'])  # Assuming 'Return' is provided in the environment info

        # Render the environment (optional)
        if render:
            env.render()

    # Calculate evaluation metrics
    total_reward = np.sum(total_rewards)
    mean_reward = np.mean(total_rewards)
    sharpe_ratio = (
        (np.mean(portfolio_returns) - risk_free_rate) / np.std(portfolio_returns)
        if len(portfolio_returns) > 1 else 0
    )
    max_drawdown = calculate_max_drawdown(portfolio_returns)
    win_rate = np.sum(np.array(total_rewards) > 0) / len(total_rewards)

    # Action distribution
    action_distribution = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}

    return {
        "Total Reward": total_reward,
        "Mean Reward": mean_reward,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Action Distribution": action_distribution,
    }

def calculate_max_drawdown(returns):
    """
    Calculate the Maximum Drawdown (MDD) from a list of returns.

    Parameters:
        returns (list): List of portfolio returns.

    Returns:
        float: Maximum Drawdown.
    """
    if len(returns) == 0:
        return 0  # Return 0 if there are no returns

    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    return max_drawdown
import matplotlib.pyplot as plt

# Evaluate the model and store the metrics
metrics = evaluate_model(env, model, risk_free_rate=0.01, render=False)

# Print evaluation metrics for debugging
print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Extract action distribution values and convert to a list
action_distribution_values = list(metrics["Action Distribution"].values())

# Calculate cumulative returns
cumulative_returns = np.cumsum(action_distribution_values)

# Plot cumulative returns
plt.plot(cumulative_returns)
plt.title("Cumulative Returns")
plt.xlabel("Time Steps")
plt.ylabel("Returns")
plt.savefig("cumulative_returns.png")
plt.show()