import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def generate_synthetic_portfolio_weights(start_date='2008-01-01', end_date='2022-12-31', 
                                        frequency='M', noise_level=0.05, trend_strength=0.03):
    """
    Generate synthetic portfolio weights for mutual funds over time with realistic dynamics.
    
    Parameters:
    - start_date: Beginning date for the time series
    - end_date: End date for the time series
    - frequency: Data frequency ('M' for monthly, 'Q' for quarterly)
    - noise_level: Amount of random variation in weights
    - trend_strength: Strength of long-term trends in allocation
    
    Returns:
    - DataFrame with dates and fund weights
    """
    # Initial target allocations for each fund (stocks, bonds)
    fund_allocations = {
        'ACEIX': [0.60, 0.40],  # Moderate allocation
        'AOVIX': [0.85, 0.15],  # Aggressive allocation
        'BERIX': [0.30, 0.70],  # Conservative allocation
        'FBALX': [0.50, 0.50],  # Balanced allocation
        'TRSGX': [0.80, 0.16],  # Growth allocation (stocks + small cash position)
        'VSCGX': [1.00, 0.00]   # 100% stock allocation
    }
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Create dataframe with dates
    df = pd.DataFrame(index=date_range)
    df.index.name = 'Date'
    
    # Generate weights for each fund
    for fund, [stock_target, bond_target] in fund_allocations.items():
        # Generate random walks with mean reversion for stock allocation
        n_periods = len(date_range)
        
        # Parameters
        mean_reversion_strength = 0.2  # How quickly it reverts to target
        cycles_per_year = 1.5  # How many market cycles in a year
        cycle_amplitude = 0.1  # Size of cyclical effect
        
        # Generate time components
        t = np.linspace(0, n_periods/12, n_periods)  # Time in years
        
        # Add cyclical component (simulates market conditions)
        cycle = cycle_amplitude * np.sin(2 * np.pi * cycles_per_year * t)
        
        # Trend component (slow changes in allocation policy)
        trend = trend_strength * np.sin(0.5 * np.pi * t / (n_periods/12))
        
        # Random noise
        noise = np.random.normal(0, noise_level, n_periods)
        
        # Combine components
        stock_weight = stock_target + cycle + trend + noise
        
        # Apply mean reversion (pull back toward target)
        for i in range(1, n_periods):
            deviation = stock_weight[i-1] - stock_target
            stock_weight[i] = stock_weight[i] - mean_reversion_strength * deviation
        
        # Ensure weights stay within reasonable bounds (not negative, not too high)
        stock_weight = np.clip(stock_weight, max(0, stock_target-0.2), min(1, stock_target+0.2))
        
        # Calculate bond weight
        bond_weight = 1.0 - stock_weight
        
        # Add to dataframe
        df[f'{fund}_stock'] = stock_weight
        df[f'{fund}_bond'] = bond_weight
    
    # Reset index to have Date as a column
    df = df.reset_index()
    
    return df

def generate_market_events_overlay(df, event_impact=0.15):
    """
    Add significant market events that would affect allocations.
    
    Parameters:
    - df: DataFrame with fund allocations
    - event_impact: Magnitude of impact on allocations
    
    Returns:
    - Modified DataFrame with event effects
    """
    # Define major market events
    events = {
        '2008-09-15': ('Financial Crisis - Lehman Bankruptcy', -event_impact),  # Reducing stock allocation
        '2009-03-09': ('Market Bottom', event_impact * 0.8),                    # Starting to increase stocks
        '2011-08-05': ('US Credit Downgrade', -event_impact * 0.5),             # Slight reduction in stocks
        '2016-06-24': ('Brexit Vote', -event_impact * 0.3),                     # Small reduction
        '2018-12-24': ('Christmas Eve Sell-off', -event_impact * 0.7),          # Larger reduction  
        '2020-03-23': ('COVID-19 Market Bottom', event_impact),                 # Increase stocks after drop
        '2022-01-03': ('Fed Rate Hike Cycle', -event_impact * 0.6)              # Reduce stocks due to rate concerns
    }
    
    # Apply event impacts
    for date_str, (description, impact) in events.items():
        event_date = pd.to_datetime(date_str)
        
        # Find closest date in our dataframe
        closest_date_idx = np.argmin(np.abs(df['Date'] - event_date))
        
        # Apply impact over next few periods (simulating gradual reaction)
        decay_periods = 3
        for i in range(decay_periods):
            if closest_date_idx + i < len(df):
                impact_factor = impact * (decay_periods - i) / decay_periods
                
                # Apply to all funds but with varying sensitivity
                for fund, sensitivity in [('ACEIX', 0.9), ('AOVIX', 0.7), ('BERIX', 1.2), 
                                        ('FBALX', 1.0), ('TRSGX', 0.8), ('VSCGX', 0.6)]:
                    
                    # Adjust stock weights
                    df.loc[closest_date_idx + i, f'{fund}_stock'] += impact_factor * sensitivity
                    
                    # Ensure weights stay within [0, 1]
                    df.loc[closest_date_idx + i, f'{fund}_stock'] = np.clip(
                        df.loc[closest_date_idx + i, f'{fund}_stock'], 0, 1)
                    
                    # Adjust bond weights to maintain sum = 1
                    df.loc[closest_date_idx + i, f'{fund}_bond'] = 1 - df.loc[closest_date_idx + i, f'{fund}_stock']
    
    return df

def plot_fund_allocations(df, output_folder='Results'):
    """
    Create visualizations of the fund allocations over time.
    
    Parameters:
    - df: DataFrame with fund allocations
    - output_folder: Where to save the plots
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot settings
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 12
    
    # 1. Plot all funds' stock allocations together
    plt.figure()
    for fund in ['ACEIX', 'AOVIX', 'BERIX', 'FBALX', 'TRSGX', 'VSCGX']:
        plt.plot(df['Date'], df[f'{fund}_stock'], label=f'{fund}')
    
    plt.title('Stock Allocation Over Time by Fund', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Allocation', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'all_funds_stock_allocation.png'), dpi=300)
    
    # 2. Create individual plots for each fund showing stock/bond allocation
    for fund in ['ACEIX', 'AOVIX', 'BERIX', 'FBALX', 'TRSGX', 'VSCGX']:
        plt.figure()
        
        plt.stackplot(df['Date'], 
                     df[f'{fund}_stock'], df[f'{fund}_bond'],
                     labels=['Stocks', 'Bonds'], 
                     colors=['#ff9999', '#66b3ff'],
                     alpha=0.8)
        
        plt.title(f'{fund} Asset Allocation Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Allocation', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{fund}_allocation.png'), dpi=300)
    
    # 3. Create a heatmap of fund strategies over time
    plt.figure(figsize=(16, 10))
    
    funds = ['ACEIX', 'AOVIX', 'BERIX', 'FBALX', 'TRSGX', 'VSCGX']
    
    # Sample dates for readability (e.g., yearly)
    sample_dates = df['Date'][::12] if len(df) > 60 else df['Date']
    sample_indices = [df[df['Date'] == date].index[0] for date in sample_dates]
    
    # Create a matrix of stock allocations
    data = np.zeros((len(funds), len(sample_dates)))
    for i, fund in enumerate(funds):
        for j, idx in enumerate(sample_indices):
            data[i, j] = df.loc[idx, f'{fund}_stock']
    
    # Plot heatmap
    plt.imshow(data, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Stock Allocation')
    
    # Add labels
    plt.yticks(range(len(funds)), funds)
    date_labels = [date.strftime('%Y-%m') for date in sample_dates]
    plt.xticks(range(len(sample_dates)), date_labels, rotation=45, ha='right')
    
    plt.title('Fund Stock Allocation Strategies Over Time', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'fund_allocation_heatmap.png'), dpi=300)
    
    plt.close('all')

def main():
    # Generate synthetic data
    synthetic_data = generate_synthetic_portfolio_weights()
    
    # Add market events effects
    synthetic_data = generate_market_events_overlay(synthetic_data)
    
    # Create visualizations
    plot_fund_allocations(synthetic_data)
    
    # Save to CSV
    output_path = os.path.join('Data', 'Processed', 'MutualFunds')
    os.makedirs(output_path, exist_ok=True)
    synthetic_data.to_csv(os.path.join(output_path, 'mutual_fund_allocations.csv'), index=False)
    
    print(f"Generated synthetic data for {len(synthetic_data)} time periods")
    print(f"Data saved to {os.path.join(output_path, 'mutual_fund_allocations.csv')}")
    print(f"Visualizations saved to Results folder")

if __name__ == "__main__":
    main()