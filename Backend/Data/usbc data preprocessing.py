import pandas as pd

def format_date_with_quarter(date_str):
    """
    Convert a date string in the format "Month Year" (e.g., "October 1860")
    into the format "October 1860 (1860Q3)".
    """
    try:
        dt = pd.to_datetime(date_str, format='%B %Y')
        quarter = dt.quarter
        return dt.strftime('%B %Y') + f" ({dt.year}Q{quarter})"
    except Exception as e:
        print(f"Error converting date '{date_str}': {e}")
        return date_str

# Load the Excel file
file_path = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/Macroeconomic Data/US Business Cycles Expansions and Contractions.xlsx"
df = pd.read_excel(file_path)

# Standardize column names by stripping extra spaces
df.columns = df.columns.str.strip()

# Print out the column names for verification
print("Columns found in the Excel file:", df.columns.tolist())

# Correct column names based on the actual column names
peak_col = "Peak month  (Peak Quarter)"  # Note the extra space
trough_col = "Trough month (Trough Quarter)"
contraction_col = "Contraction"
expansion_col = "Expansion"
cycle_trough_col = "Cycle (Trough to Trough)"
cycle_peak_col = "Cycle 2 (Peak to Peak)"

# Process the date columns using the formatting function
df[peak_col] = df[peak_col].str.extract(r"([A-Za-z]+ \d{4})")[0].apply(format_date_with_quarter)
df[trough_col] = df[trough_col].str.extract(r"([A-Za-z]+ \d{4})")[0].apply(format_date_with_quarter)

# Convert numeric columns to proper format
numeric_columns = [contraction_col, expansion_col, cycle_trough_col, cycle_peak_col]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values in the relevant columns
df.dropna(
    subset=[peak_col, trough_col, contraction_col, expansion_col, cycle_trough_col, cycle_peak_col],
    inplace=True,
)

# Save the formatted DataFrame to a new CSV file
output_filename = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data/formatted_usbc_data.csv"
df.to_csv(output_filename, index=False)

print(f"Data has been formatted and saved to '{output_filename}'.")