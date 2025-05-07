import os
import pandas as pd

# Path to the folder containing the Excel files
data_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data"

# List all Excel files in the folder
files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]

# Iterate through each Excel file
for file in files:
    file_path = os.path.join(data_folder, file)
    print(f"\nProcessing File: {file}")
    
    # Load the Excel file
    excel_data = pd.ExcelFile(file_path)
    
    # Display the sheet names
    print("Sheet Names:", excel_data.sheet_names)
    
    # Iterate through each sheet in the file
    for sheet_name in excel_data.sheet_names:
        print(f"\n  Analyzing Sheet: {sheet_name}")
        # Load the sheet into a DataFrame
        df = excel_data.parse(sheet_name)
        
        # Check if the DataFrame is empty
        if df.empty:
            print("  The sheet is empty. Skipping...")
            continue
        
        # Display the first few rows
        print("  First 5 rows:")
        print(df.head())
        
        # Display basic statistics
        print("\n  Basic Statistics:")
        print(df.describe())
        
        # Check for missing values
        print("\n  Missing Values:")
        print(df.isnull().sum())