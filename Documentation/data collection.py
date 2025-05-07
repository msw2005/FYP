import yfinance as yf
import pandas as pd
import os

# Define mutual funds and their categories
mutual_funds = {
    "Aggressive Allocation": ["TRSGX", "AOVIX"],
    "Moderate Allocation": ["FBALX", "ACEIX"],
    "Conservative Allocation": ["VSCGX", "BERIX"]
}

# Define the data collection period
start_date = "2008-01-01"
end_date = "2022-12-31"

# Directory to save the data
output_dir = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/MutualFunds"
os.makedirs(output_dir, exist_ok=True)

# Function to fetch and save data for each mutual fund
def fetch_mutual_fund_data(fund_symbol, category):
    print(f"Fetching data for {fund_symbol} ({category})...")
    try:
        # Fetch historical data
        fund_data = yf.download(fund_symbol, start=start_date, end=end_date, progress=False)
        
        # Keep only the 'Close' column and reset the index
        fund_data = fund_data[['Close']].reset_index()
        
        # Rename columns for clarity
        fund_data.rename(columns={"Date": "Date", "Close": "MutualFunds_Close"}, inplace=True)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{fund_symbol}_{category}.csv")
        fund_data.to_csv(output_file, index=False, date_format="%Y-%m-%d")
        print(f"Data for {fund_symbol} saved to {output_file}.")
    except Exception as e:
        print(f"Failed to fetch data for {fund_symbol}: {e}")

# Iterate over mutual funds and fetch data
for category, funds in mutual_funds.items():
    for fund in funds:
        fetch_mutual_fund_data(fund, category)

# Combine all data into a single CSV (optional)
def combine_data():
    combined_data = pd.DataFrame()
    for file in os.listdir(output_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(output_dir, file)
            fund_data = pd.read_csv(file_path)
            fund_data["Fund"] = file.split("_")[0]  # Add fund symbol as a column
            combined_data = pd.concat([combined_data, fund_data])
    combined_output_file = os.path.join(output_dir, "combined_mutual_fund_data.csv")
    combined_data.to_csv(combined_output_file, index=False)
    print(f"Combined data saved to {combined_output_file}.")

combine_data()