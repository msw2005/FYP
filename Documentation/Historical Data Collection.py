import yfinance as yf
import pandas as pd
import os

# Define the assets and their symbols
assets = {
    "Equity Market (S&P 500)": "^GSPC",  # S&P 500 Index
    "Fixed-Income (FTSE USBIG)": "USBIG"  # FTSE US Broad Investment-Grade Bond Index (symbol may vary)
}

# Define the data collection period
start_date = "2008-01-01"
end_date = "2022-12-31"

# Directory to save the data
output_dir = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/MarketData"
os.makedirs(output_dir, exist_ok=True)

# Function to fetch and save data for each asset
def fetch_market_data(asset_name, symbol):
    print(f"Fetching data for {asset_name} ({symbol})...")
    try:
        # Fetch historical data
        asset_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{asset_name.replace(' ', '_')}.csv")
        asset_data.to_csv(output_file)
        print(f"Data for {asset_name} saved to {output_file}.")
    except Exception as e:
        print(f"Failed to fetch data for {asset_name}: {e}")

# Iterate over assets and fetch data
for asset_name, symbol in assets.items():
    fetch_market_data(asset_name, symbol)

# Combine all data into a single CSV (optional)
def combine_data():
    combined_data = pd.DataFrame()
    for file in os.listdir(output_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(output_dir, file)
            asset_data = pd.read_csv(file_path)
            asset_data["Asset"] = file.split(".")[0]  # Add asset name as a column
            combined_data = pd.concat([combined_data, asset_data])
    combined_output_file = os.path.join(output_dir, "combined_market_data.csv")
    combined_data.to_csv(combined_output_file, index=False)
    print(f"Combined data saved to {combined_output_file}.")

combine_data()