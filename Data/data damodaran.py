import os
import pandas as pd

def process_damodaran_data(input_file, output_folder):
    """
    Process the Damodaran dataset and save cleaned data to the output folder.

    Parameters:
        input_file (str): Path to the raw Damodaran Excel file.
        output_folder (str): Path to save the processed datasets.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the Excel file
    print(f"Loading Damodaran data from {input_file}...")
    excel_data = pd.ExcelFile(input_file)

    # Iterate through each sheet
    for sheet_name in excel_data.sheet_names:
        print(f"Processing sheet: {sheet_name}...")
        try:
            # Load the sheet into a DataFrame
            sheet_data = excel_data.parse(sheet_name)

            # Drop completely empty rows and columns
            sheet_data.dropna(how="all", inplace=True)
            sheet_data.dropna(axis=1, how="all", inplace=True)

            # Rename columns for clarity (if applicable)
            sheet_data.rename(columns=lambda x: str(x).strip(), inplace=True)

            # Handle missing values (e.g., forward-fill or drop rows with critical missing data)
            sheet_data.fillna(method="ffill", inplace=True)

            # Save the processed sheet to a CSV file
            output_file = os.path.join(output_folder, f"{sheet_name.replace(' ', '_')}.csv")
            sheet_data.to_csv(output_file, index=False)
            print(f"Saved processed data for sheet '{sheet_name}' to {output_file}.")
        except Exception as e:
            print(f"Failed to process sheet '{sheet_name}': {e}")

# Example usage
if __name__ == "__main__":
    input_file = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Damodaran- Expected Returns/histretSP.xlsx"
    output_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed/Damodaran"
    process_damodaran_data(input_file, output_folder)