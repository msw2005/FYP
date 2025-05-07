import pandas as pd

# Define file paths
input_file = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed/Damodaran/Returns_by_year.csv"
output_file = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Processed/Damodaran/Refined_Returns_by_year.csv"

# Load the dataset
data = pd.read_csv(input_file)

# Ensure the required columns are present
required_columns = ["Year", "Historical ERP", "3-month T.Bill"]
if not all(column in data.columns for column in required_columns):
    raise ValueError(f"Missing one or more required columns: {required_columns}")

# Filter the dataset to include only the required columns
refined_data = data[required_columns]

# Save the refined dataset to a new CSV file
refined_data.to_csv(output_file, index=False)

print(f"Refined dataset saved to {output_file}")