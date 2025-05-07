import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define data folder
data_folder = "/Users/alisadiq/Library/CloudStorage/OneDrive-UniversityofGreenwich/Documents/Year 3/FYP/Data"

# Load datasets
sp500_data = pd.read_csv(os.path.join(data_folder, "MarketData/Equity_Market_(S&P_500).csv"))
bond_data = pd.read_csv(os.path.join(data_folder, "MarketData/ICE US Bonds Index Yield.csv"))
bond_data.rename(columns={"observation_date": "Date"}, inplace=True)
inflation_data = pd.read_excel(os.path.join(data_folder, "Macroeconomic Data/12-month percentage change, Consumer Price Index, selected categories, not seasonally adjusted.xlsx"))
unemployment_data = pd.read_excel(os.path.join(data_folder, "Macroeconomic Data/Labor Force Statistics from the Current Population Survey.xlsx"))
risk_free_rate_data = pd.read_csv(os.path.join(data_folder, "Macroeconomic Data/Market Yield on U.S. Treasury Securities at 1-Month Constant Maturity, Quoted on an Investment Basis.csv"))
business_cycle_data = pd.read_excel(os.path.join(data_folder, "Macroeconomic Data/US Business Cycles Expansions and Contractions.xlsx"))

# 1. Basic Information
print("S&P 500 Data Info:")
print(sp500_data.info())
print("\nBond Data Info:")
print(bond_data.info())
print("\nInflation Data Info:")
print(inflation_data.info())
print("\nUnemployment Data Info:")
print(unemployment_data.info())
print("\nRisk-Free Rate Data Info:")
print(risk_free_rate_data.info())
print("\nBusiness Cycle Data Info:")
print(business_cycle_data.info())

# 2. Check for Missing Values
print("\nMissing Values in S&P 500 Data:")
print(sp500_data.isnull().sum())
print("\nMissing Values in Bond Data:")
print(bond_data.isnull().sum())
print("\nMissing Values in Inflation Data:")
print(inflation_data.isnull().sum())
print("\nMissing Values in Unemployment Data:")
print(unemployment_data.isnull().sum())
print("\nMissing Values in Risk-Free Rate Data:")
print(risk_free_rate_data.isnull().sum())
print("\nMissing Values in Business Cycle Data:")
print(business_cycle_data.isnull().sum())

# 3. Descriptive Statistics
print("\nS&P 500 Data Statistics:")
print(sp500_data.describe())
print("\nBond Data Statistics:")
print(bond_data.describe())
print("\nInflation Data Statistics:")
print(inflation_data.describe())
print("\nUnemployment Data Statistics:")
print(unemployment_data.describe())
print("\nRisk-Free Rate Data Statistics:")
print(risk_free_rate_data.describe())

# 4. Visualizations

# Plot S&P 500 Prices
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(sp500_data['Date']), sp500_data['Close'], label='S&P 500 Close Price')
plt.title("S&P 500 Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Plot Bond Yields
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(bond_data['Date']), bond_data['BAMLC0A0CMEY'], label='Bond Yield')
plt.title("Bond Yield Over Time")
plt.xlabel("Date")
plt.ylabel("Yield")
plt.legend()
plt.show()

# Correlation Heatmap for S&P 500 and Bond Data
merged_data = sp500_data.merge(bond_data, on="Date", how="inner")
correlation_matrix = merged_data[['Close', 'BAMLC0A0CMEY']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Between S&P 500 and Bond Yield")
plt.show()

# Plot Inflation Over Time
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(inflation_data['Month']), inflation_data['All items'], label='Inflation')
plt.title("Inflation Over Time")
plt.xlabel("Date")
plt.ylabel("CPI")
plt.legend()
plt.show()

# Reshape unemployment data to long format
unemployment_data_long = unemployment_data.melt(id_vars=['Year'], var_name='Month', value_name='Unemployment Rate')

# Combine Year and Month into a single datetime column
unemployment_data_long['Date'] = pd.to_datetime(unemployment_data_long['Year'].astype(str) + '-' + unemployment_data_long['Month'])

# Fix unemployment data parsing
month_mapping = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
unemployment_data_long['Month'] = unemployment_data_long['Month'].map(month_mapping)
unemployment_data_long['Date'] = pd.to_datetime(unemployment_data_long['Year'].astype(str) + '-' + unemployment_data_long['Month'] + '-01')

# Fix risk-free rate data parsing
risk_free_rate_data['observation_date'] = pd.to_datetime(risk_free_rate_data['observation_date'], format='%d/%m/%Y', errors='coerce')

# Plot Unemployment Over Time
plt.figure(figsize=(10, 6))
plt.plot(unemployment_data_long['Date'], unemployment_data_long['Unemployment Rate'], label='Unemployment Rate')
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate")
plt.legend()
plt.show()

# Plot Risk-Free Rate Over Time
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(risk_free_rate_data['observation_date']), risk_free_rate_data['DGS1MO'], label='Risk-Free Rate')
plt.title("Risk-Free Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.legend()
plt.show()

# Preprocess US Business Cycle Data
import pandas as pd
import matplotlib.pyplot as plt

# Example dataset
data = {
    'Peak month  (Peak Quarter)': ['October 1860 (1860Q3)', 'June 1857 (1857Q2)'],
    'Trough month (Trough Quarter)': ['June 1861 (1861Q3)', 'December 1858 (1858Q4)'],
    'Contraction': [8, 18],
    'Expansion': [22, 30],
    'Cycle': ['Cycle 1', 'Cycle 2']
}

business_cycle_data = pd.DataFrame(data)

# Extract and parse dates
business_cycle_data['Peak month  (Peak Quarter)'] = pd.to_datetime(
    business_cycle_data['Peak month  (Peak Quarter)'].str.extract(r'([A-Za-z]+ \d{4})')[0], format='%B %Y', errors='coerce'
)
business_cycle_data['Trough month (Trough Quarter)'] = pd.to_datetime(
    business_cycle_data['Trough month (Trough Quarter)'].str.extract(r'([A-Za-z]+ \d{4})')[0], format='%B %Y', errors='coerce'
)

# Drop rows with missing values in relevant columns
business_cycle_data.dropna(subset=['Peak month  (Peak Quarter)', 'Trough month (Trough Quarter)', 'Contraction', 'Expansion'], inplace=True)

# Plot US Business Cycle Data
plt.figure(figsize=(12, 8))
for i, row in business_cycle_data.iterrows():
    # Plot contraction phase
    plt.barh(
        i,
        row['Contraction'],
        left=row['Peak month  (Peak Quarter)'].toordinal(),
        color='red',
        alpha=0.7,
        label='Contraction' if i == 0 else ""
    )
    # Plot expansion phase
    plt.barh(
        i,
        row['Expansion'],
        left=row['Trough month (Trough Quarter)'].toordinal(),
        color='blue',
        alpha=0.7,
        label='Expansion' if i == 0 else ""
    )

# Format the plot
plt.title("US Business Cycle Expansions and Contractions")
plt.xlabel("Year")
plt.ylabel("Cycle Index")
plt.yticks(range(len(business_cycle_data)), business_cycle_data['Cycle'])
plt.legend(["Contraction", "Expansion"])
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()