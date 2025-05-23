Portfolio Allocation System with Deep Reinforcement Learning
Project Overview
This project implements a sophisticated portfolio allocation system that compares traditional Mean-Variance Optimization (MVO) with advanced Deep Reinforcement Learning (RL) approaches. The system processes historical financial data, builds and trains portfolio allocation models, and presents results through an interactive web interface, allowing users to explore different risk preferences and allocation strategies.

System Architecture
The system follows a modular architecture divided into several components:

Data Processing Pipeline: Handles acquisition and preprocessing of financial data from various sources
Model Implementation: Contains MVO, Inverse Optimization, and Deep RL implementations
Web Backend: Flask API serving model results and handling model execution
Web Frontend: Interactive dashboard for model visualization and control
Directory Structure
├── Backend/
│   ├── app.py               # Main Flask application
│   ├── requirements.txt     # Python dependencies
│   ├── Data/                # Processed data for backend use
│   ├── Models/              # Model implementations
│   ├── static/              # Static assets (CSS, JS)
│   ├── templates/           # HTML templates
│   └── utils/               # Helper functions
├── Data/
│   ├── data damodaran.py    # Damodaran dataset processing scripts
│   ├── data pre-process.py  # Main data preprocessing scripts
│   ├── EDA.py               # Exploratory data analysis
│   ├── Synthetic data.py    # Synthetic data generation scripts
│   └── formatted_usbc_data.csv  # Processed business cycle data
├── Documentation/
│   ├── Algorithms.txt       # Detailed algorithm descriptions
│   ├── Datasets Descriptions  # Documentation of data sources
│   └── Readme.txt           # General setup instructions
├── Frontend/
│   ├── styling.css          # CSS for dashboard
│   └── frontend.js          # JavaScript for interactive features
├── Models/
│   ├── models.ipynb         # Jupyter notebook with model implementations
│   └── Deep RL1/            # Saved Deep RL models
├── Results/                 # Output files from model runs
└── Testing/
    └── usability/           # Usability testing scripts and documentation
    
Key Components
Data Processing
The system handles multiple data sources including:

S&P 500 index data
US Bond index data
Risk-free rate data
Damodaran expected returns data
Business cycle data
Synthetic mutual fund allocation data
The preprocessing pipeline includes:

Date standardization across sources
Missing value handling with forward-fill techniques
Feature engineering (volatility measures, moving averages, RSI)
Normalization using RobustScaler
Model Implementations
Mean-Variance Optimization (MVO)
Traditional Markowitz portfolio optimization that:

Estimates expected returns and covariance from historical data
Uses sequential least squares programming to maximize utility
Allows risk aversion parameter adjustment
Inverse Optimization
Determines which risk aversion parameter would generate observed allocations by:

Taking observed portfolio weights from mutual funds
Finding risk aversion that minimizes difference between model and observed weights
Providing calibration for other models
Deep Reinforcement Learning
Advanced portfolio allocation using PPO (Proximal Policy Optimization):

Custom Gym environment for portfolio simulation
Neural network architecture with separate policy and value networks
Reward function balancing returns against risk metrics
Tensorboard logging for performance monitoring
Web Interface
Interactive dashboard featuring:

Portfolio performance metrics display
Asset allocation visualization over time
Model comparison tools
Parameter adjustment controls
Real-time data updates
Export functionality for results

Setup & Installation
pip install -r Backend/requirements.txt

Prerequisites
Python 3.10+
PyTorch
Flask
Stable-Baselines3
NumPy, Pandas, Matplotlib
Gymnasium
Installation Steps
Clone the repository

Install dependencies:

Update path configurations in:
# For Windows
BASE_DIR = r"c:\path\to\project"
# For macOS/Linux
BASE_DIR = r"/path/to/project"

app.py
models.ipynb
Replace the base directory paths with your local paths:

Ensure the data directory structure matches the expected paths or update paths accordingly

Running the Application

Start the Flask backend:
cd Backend
python app.py

Open a web browser and navigate to:
http://localhost:5000

Using the System
View Portfolio Metrics: The dashboard displays key portfolio performance metrics including annualized returns, Sharpe ratio, volatility, and maximum drawdown.

Run Mean-Variance Optimization:

Select the MVO model option
Adjust the risk aversion parameter (default 3.0)
Click "Run Model" and observe the results
Run Deep Reinforcement Learning:

Select the Deep RL model option
Set a low risk aversion parameter (0.1-0.5) for better returns
Click "Run Model" (this may take several minutes)
Compare Model Performance:

The Model Comparison table shows side-by-side metrics
Export results to CSV for further analysis
Analyze Allocations:

View the allocation chart showing stock vs bond weights
See how allocations change over different time periods
Technical Notes
Deep RL Training: The system uses temporary Python scripts to execute Deep RL training, which may take several minutes to complete.

Risk Aversion Parameter: Lower values (0.1-0.5) focus on returns, while higher values (3+) prioritize risk reduction.

Path Management: The system uses a flexible path resolution system to locate data and model files across different operating systems.

Data Requirements: Ensure all required data files are available in the specified Data/Processed directory.

Project Features
Comparison of traditional vs. AI-based portfolio allocation methods
Interactive risk parameter adjustment
Historical backtesting of allocation strategies
Visual performance comparison between models
Export functionality for further analysis
