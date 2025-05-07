import os
import pandas as pd
import numpy as np

def process_and_save_data(data_folder, output_folder):
    """
    Process all datasets, clean them, and save the processed data to new files.

    Parameters:
        data_folder (str): Path to the folder containing the raw datasets.
        output_folder (str): Path to the folder where processed datasets will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load datasets
    sp500_data = pd.read_csv(os.path.join(data_folder, "MarketData/Equity_Market_(S&P_500).csv"))
    bond_data = pd.read_csv(os.path.join(data_folder, "MarketData/ICE US Bonds Index Yield.csv"))
    inflation_data = pd.read_excel(os.path.join(data_folder, "Macroeconomic Data/12-month percentage change, Consumer Price Index, selected categories, not seasonally adjusted.xlsx"))
    unemployment_data = pd.read_excel(os.path.join(data_folder, "Macroeconomic Data/Labor Force Statistics from the Current Population Survey.xlsx"))
    risk_free_rate_data = pd.read_csv(os.path.join(data_folder, "Macroeconomic Data/Market Yield on U.S. Treasury Securities at 1-Month Constant Maturity, Quoted on an Investment Basis.csv"))
    business_cycle_data = pd.read_csv(os.path.join(data_folder, "formatted_usbc_data.csv"))

    # Rename columns for consistency
    bond_data.rename(columns={"observation_date": "Date"}, inplace=True)
    risk_free_rate_data.rename(columns={"observation_date": "Date"}, inplace=True)
    sp500_data.rename(columns={"Close": "SP500_Close"}, inplace=True)

    # Convert date columns to datetime format (standardized to YYYY-MM-DD)
    sp500_data["Date"] = pd.to_datetime(sp500_data["Date"], errors="coerce")
    bond_data["Date"] = pd.to_datetime(bond_data["Date"], errors="coerce")
    inflation_data["Month"] = pd.to_datetime(inflation_data["Month"], errors="coerce")
    
    # Fix date parsing for risk-free rate data (DD/MM/YYYY to YYYY-MM-DD)
    risk_free_rate_data["Date"] = pd.to_datetime(risk_free_rate_data["Date"], format="%d/%m/%Y", errors="coerce")
    
    # Convert business cycle dates
    business_cycle_data["Peak month  (Peak Quarter)"] = pd.to_datetime(
        business_cycle_data["Peak month  (Peak Quarter)"].str.extract(r"([A-Za-z]+ \d{4})")[0], format="%B %Y", errors="coerce"
    )
    business_cycle_data["Trough month (Trough Quarter)"] = pd.to_datetime(
        business_cycle_data["Trough month (Trough Quarter)"].str.extract(r"([A-Za-z]+ \d{4})")[0], format="%B %Y", errors="coerce"
    )

    # Process unemployment data
    unemployment_data_long = unemployment_data.melt(id_vars=["Year"], var_name="Month", value_name="Unemployment Rate")
    month_mapping = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                     'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    unemployment_data_long["Month"] = unemployment_data_long["Month"].map(month_mapping)
    unemployment_data_long["Date"] = pd.to_datetime(unemployment_data_long["Year"].astype(str) + '-' + unemployment_data_long["Month"] + '-01')

    # Drop rows with missing values in critical columns
    sp500_data.dropna(subset=["SP500_Close"], inplace=True)
    bond_data.dropna(subset=["BAMLC0A0CMEY"], inplace=True)
    inflation_data.dropna(subset=["All items"], inplace=True)
    unemployment_data_long.dropna(subset=["Unemployment Rate"], inplace=True)
    risk_free_rate_data.dropna(subset=["DGS1MO"], inplace=True)
    business_cycle_data.dropna(subset=["Peak month  (Peak Quarter)", "Trough month (Trough Quarter)", "Contraction", "Expansion"], inplace=True)

    # Save processed datasets to new files
    sp500_data.to_csv(os.path.join(output_folder, "processed_sp500_data.csv"), index=False, date_format="%Y-%m-%d")
    bond_data.to_csv(os.path.join(output_folder, "processed_bond_data.csv"), index=False, date_format="%Y-%m-%d")
    inflation_data.to_csv(os.path.join(output_folder, "processed_inflation_data.csv"), index=False, date_format="%Y-%m-%d")
    unemployment_data_long.to_csv(os.path.join(output_folder, "processed_unemployment_data.csv"), index=False, date_format="%Y-%m-%d")
    risk_free_rate_data.to_csv(os.path.join(output_folder, "processed_risk_free_rate_data.csv"), index=False, date_format="%Y-%m-%d")
    business_cycle_data.to_csv(os.path.join(output_folder, "processed_business_cycle_data.csv"), index=False, date_format="%Y-%m-%d")

    print("All datasets have been processed and saved to:", output_folder)

# Example usage
if __name__ == "__main__":
    data_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data"
    output_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed"
    process_and_save_data(data_folder, output_folder)