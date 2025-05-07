import os
import pandas as pd

def preprocess_mutual_funds_data(input_folder, output_folder):
    """
    Preprocess individual mutual funds datasets and save them to a new folder.

    Parameters:
        input_folder (str): Path to the folder containing the raw mutual funds datasets.
        output_folder (str): Path to the folder where processed datasets will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all CSV files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_name}...")

            # Load the dataset, skipping the second row (index 1)
            data = pd.read_csv(file_path, skiprows=[1])

            # Rename columns for consistency
            data.rename(columns={"Date": "Date", "MutualFunds_Close": "MutualFunds_Close"}, inplace=True)

            # Convert the Date column to datetime format
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

            # Drop rows with missing or invalid dates
            data.dropna(subset=["Date"], inplace=True)

            # Drop rows with missing values in the MutualFunds_Close column
            data.dropna(subset=["MutualFunds_Close"], inplace=True)

            # Save the processed dataset to the output folder
            output_file_path = os.path.join(output_folder, file_name)
            data.to_csv(output_file_path, index=False, date_format="%Y-%m-%d")
            print(f"Saved processed data to {output_file_path}")

    print("All mutual funds datasets have been processed and saved.")

# Example usage
if __name__ == "__main__":
    input_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/MutualFunds"
    output_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed/MutualFunds"
    preprocess_mutual_funds_data(input_folder, output_folder)