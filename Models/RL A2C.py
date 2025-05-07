from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import gym

# 1. Preprocess the MutualFunds data

def preprocess_mutual_funds_data(data_folder):
    """
    Preprocess MutualFunds datasets by combining historical market data and technical indicators.
    """
    # Load MutualFunds data from the Processed folder
    mutual_funds_data = pd.read_csv(
        os.path.join(data_folder, "Processed/MutualFunds/combined_mutual_fund_data.csv")
    )
    mutual_funds_data.rename(columns={"Date": "Date", "MutualFunds_Close": "MutualFunds_Close"}, inplace=True)

    # Convert date column to datetime format
    mutual_funds_data["Date"] = pd.to_datetime(mutual_funds_data["Date"], errors="coerce")

    # Drop rows with missing values
    mutual_funds_data.dropna(subset=["MutualFunds_Close"], inplace=True)

    # Ensure the DataFrame is not empty
    if mutual_funds_data.empty:
        raise ValueError("The preprocessed MutualFunds data is empty. Please check the input dataset.")

    # Normalize the MutualFunds_Close column
    scaler = MinMaxScaler()
    mutual_funds_data["MutualFunds_Close"] = scaler.fit_transform(mutual_funds_data[["MutualFunds_Close"]])

    return mutual_funds_data, scaler

# 2. Define the custom environment

class MutualFundsEnv(gym.Env):
    """
    Custom Environment for Deep RL MutualFunds Prediction.
    """
    def __init__(self, data):
        super(MutualFundsEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # Define action space (predict the next value)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Define observation space (state space)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        return np.array([self.data.iloc[self.current_step]["MutualFunds_Close"]])

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False

        actual_value = self.data.iloc[self.current_step]["MutualFunds_Close"]
        reward = -abs(action[0] - actual_value)  # Reward is negative absolute error
        info = {"actual_value": actual_value, "predicted_value": action[0]}

        return np.array([actual_value]), reward, done, info

    def render(self, mode='human'):
        pass

# 3. Train the RL Agent

# Define paths
data_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data"

# Preprocess the MutualFunds dataset
mutual_funds_data, scaler = preprocess_mutual_funds_data(data_folder)

# Split data into training and testing sets
train_data = mutual_funds_data[mutual_funds_data["Date"] < "2020-01-01"]
test_data = mutual_funds_data[mutual_funds_data["Date"] >= "2020-01-01"]

# Create the custom environment
env = DummyVecEnv([lambda: MutualFundsEnv(train_data)])

# Define the RL model (A2C)
model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0001, gamma=0.99)

# Train the model
model.learn(total_timesteps=200000)

# Save the trained model
model_path = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Models/a2c_mutual_funds_model"
model.save(model_path)

# 4. Evaluate the RL Agent

# Load the trained model
model = A2C.load(model_path)

# Create the testing environment
test_env = MutualFundsEnv(test_data)
obs = test_env.reset()
actual_values = []
predicted_values = []

# Evaluate the model
for _ in range(len(test_data) - 1):
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    actual_values.append(info["actual_value"])
    predicted_values.append(info["predicted_value"])
    if done:
        break

# Rescale the predicted and actual values back to the original scale
actual_values = scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))
predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

# 5. Visualize the results

# Ensure the Results directory exists
results_dir = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Results"
os.makedirs(results_dir, exist_ok=True)

# Plot and save the results
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label="Actual Values", color="blue")
plt.plot(predicted_values, label="Predicted Values", color="red")
plt.title("MutualFunds Prediction vs Actual Values")
plt.xlabel("Time Steps")
plt.ylabel("MutualFunds Close Price")
plt.legend()
plt.savefig(os.path.join(results_dir, "mutual_funds_prediction.png"))
plt.show()