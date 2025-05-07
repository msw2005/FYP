import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.path import Path
import os

def create_data_preprocessing_visualization(save_path="data_preprocessing_pipeline.png"):
    """
    Create a visualization of the data preprocessing pipeline for the portfolio allocation system.
    
    Args:
        save_path (str): Path to save the visualization.
    """
    # Set up the figure with high resolution
    plt.figure(figsize=(16, 12), dpi=300)
    
    # Create a grid layout
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1.5, 1.5, 1])
    
    # Define colors
    colors = {
        'market_data': '#3498db',        # Blue
        'macro_data': '#2ecc71',         # Green
        'synthetic_data': '#e74c3c',     # Red
        'processed_data': '#9b59b6',     # Purple
        'feature_eng': '#f39c12',        # Orange
        'model_data': '#1abc9c',         # Teal
        'mvo': '#34495e',                # Dark blue
        'deep_rl': '#e67e22',            # Dark orange
        'background': '#f8f9fa',         # Light gray
        'border': '#343a40',             # Dark gray
        'text': '#2c3e50',               # Very dark blue
        'arrow': '#95a5a6'               # Gray
    }
    
    # Set background color
    plt.gcf().set_facecolor(colors['background'])
    
    # --------------------------
    # 1. Top Section: Data Sources
    # --------------------------
    data_sources_ax = plt.subplot(gs[0, :])
    data_sources_ax.set_facecolor(colors['background'])
    
    # Title for data sources section
    data_sources_ax.text(0.5, 0.9, 'Data Sources', 
                      fontsize=20, fontweight='bold', color=colors['text'],
                      horizontalalignment='center')
    
    # Create boxes for each data source type
    # Market Data
    market_rect = patches.Rectangle((0.1, 0.2), 0.2, 0.5, 
                                 linewidth=2, edgecolor=colors['market_data'], 
                                 facecolor=colors['market_data'], alpha=0.2, 
                                 label='Market Data')
    data_sources_ax.add_patch(market_rect)
    data_sources_ax.text(0.2, 0.5, 'Market Data', 
                      fontsize=14, color=colors['text'], 
                      horizontalalignment='center')
    data_sources_ax.text(0.2, 0.35, 'S&P 500\nUS Bond Index\nRisk-Free Rate', 
                      fontsize=10, color=colors['text'], 
                      horizontalalignment='center')
    
    # Macroeconomic Data
    macro_rect = patches.Rectangle((0.4, 0.2), 0.2, 0.5, 
                                linewidth=2, edgecolor=colors['macro_data'], 
                                facecolor=colors['macro_data'], alpha=0.2, 
                                label='Macroeconomic Data')
    data_sources_ax.add_patch(macro_rect)
    data_sources_ax.text(0.5, 0.5, 'Macroeconomic Data', 
                      fontsize=14, color=colors['text'], 
                      horizontalalignment='center')
    data_sources_ax.text(0.5, 0.35, 'Inflation\nUnemployment\nBusiness Cycles', 
                      fontsize=10, color=colors['text'], 
                      horizontalalignment='center')
    
    # Synthetic/Historical Data
    synth_rect = patches.Rectangle((0.7, 0.2), 0.2, 0.5, 
                                linewidth=2, edgecolor=colors['synthetic_data'], 
                                facecolor=colors['synthetic_data'], alpha=0.2, 
                                label='Synthetic/Historical Data')
    data_sources_ax.add_patch(synth_rect)
    data_sources_ax.text(0.8, 0.5, 'Synthetic Data', 
                      fontsize=14, color=colors['text'], 
                      horizontalalignment='center')
    data_sources_ax.text(0.8, 0.35, 'Mutual Fund Allocations\nDamodaran Returns', 
                      fontsize=10, color=colors['text'], 
                      horizontalalignment='center')
    
    # Remove axes
    data_sources_ax.set_xticks([])
    data_sources_ax.set_yticks([])
    data_sources_ax.spines['top'].set_visible(False)
    data_sources_ax.spines['right'].set_visible(False)
    data_sources_ax.spines['bottom'].set_visible(False)
    data_sources_ax.spines['left'].set_visible(False)
    
    # --------------------------
    # 2. Middle Left: Data Cleaning and Initial Processing
    # --------------------------
    preprocessing_ax = plt.subplot(gs[1, 0:2])
    preprocessing_ax.set_facecolor(colors['background'])
    
    # Title for preprocessing section
    preprocessing_ax.text(0.5, 0.95, 'Data Cleaning & Preprocessing', 
                       fontsize=16, fontweight='bold', color=colors['text'],
                       horizontalalignment='center')
    
    # Create preprocessing steps boxes
    steps = [
        {'name': 'Date Standardization', 'desc': 'Convert all date formats\nto YYYY-MM-DD'},
        {'name': 'Missing Value Handling', 'desc': 'Forward-fill, backward-fill\nReplace NaNs where appropriate'},
        {'name': 'Merge Datasets', 'desc': 'Join datasets on date\nInner join to ensure alignment'},
        {'name': 'Handle Outliers', 'desc': 'Clip extreme values\nRemove data errors'},
        {'name': 'Calculate Returns', 'desc': 'Price to returns conversion\nPercentage changes'}
    ]
    
    # Place preprocessing steps in boxes
    y_positions = np.linspace(0.8, 0.1, len(steps))
    for i, step in enumerate(steps):
        preprocess_rect = patches.Rectangle((0.1, y_positions[i]-0.06), 0.8, 0.12, 
                                          linewidth=2, edgecolor=colors['processed_data'], 
                                          facecolor=colors['processed_data'], alpha=0.2)
        preprocessing_ax.add_patch(preprocess_rect)
        preprocessing_ax.text(0.3, y_positions[i], step['name'], 
                           fontsize=12, color=colors['text'], 
                           horizontalalignment='center', verticalalignment='center')
        preprocessing_ax.text(0.7, y_positions[i], step['desc'], 
                           fontsize=10, color=colors['text'], 
                           horizontalalignment='center', verticalalignment='center',
                           style='italic')
    
    # Add code snippet for preprocessing
    code_box = patches.Rectangle((0.86, 0.1), 0.12, 0.8, 
                              linewidth=1, edgecolor=colors['border'], 
                              facecolor='white', alpha=0.8)
    preprocessing_ax.add_patch(code_box)
    preprocessing_ax.text(0.92, 0.85, 'Python Code', 
                       fontsize=8, color=colors['text'], 
                       horizontalalignment='center', verticalalignment='center')
    
    code_snippet = """
# Convert date columns
df['Date'] = pd.to_datetime(
    df['Date'], errors='coerce')

# Handle missing values
df = df.ffill().bfill()

# Calculate returns
df['SP500_Return'] = df['SP500']
    .pct_change()

# Clip extreme values
df['SP500_Return'] = df[
    'SP500_Return'].clip(-0.1, 0.1)
    """
    preprocessing_ax.text(0.92, 0.5, code_snippet, 
                       fontsize=6, color=colors['text'], 
                       horizontalalignment='center', verticalalignment='center',
                       family='monospace')
    
    # Remove axes
    preprocessing_ax.set_xticks([])
    preprocessing_ax.set_yticks([])
    preprocessing_ax.spines['top'].set_visible(False)
    preprocessing_ax.spines['right'].set_visible(False)
    preprocessing_ax.spines['bottom'].set_visible(False)
    preprocessing_ax.spines['left'].set_visible(False)
    
    # --------------------------
    # 3. Middle Right: Time Series Plot Sample
    # --------------------------
    sample_plot_ax = plt.subplot(gs[1, 2])
    sample_plot_ax.set_facecolor(colors['background'])
    
    # Generate sample time series data
    np.random.seed(42)  # For reproducibility
    x = np.linspace(0, 1, 100)
    sp500 = np.cumsum(np.random.normal(0.001, 0.02, 100))
    bonds = np.cumsum(np.random.normal(0.0005, 0.005, 100))
    
    # Plot the data
    sample_plot_ax.plot(x, sp500, label='S&P 500', color=colors['market_data'], linewidth=2)
    sample_plot_ax.plot(x, bonds, label='US Bonds', color=colors['macro_data'], linewidth=2)
    
    # Add title and labels
    sample_plot_ax.set_title('Processed Market Data Sample', fontsize=12, color=colors['text'])
    sample_plot_ax.set_xlabel('Time', fontsize=10, color=colors['text'])
    sample_plot_ax.set_ylabel('Value', fontsize=10, color=colors['text'])
    sample_plot_ax.legend(loc='upper left', frameon=True, fontsize=8)
    
    # Add date ranges
    sample_plot_ax.axvline(x=0.75, color=colors['border'], linestyle='--')
    sample_plot_ax.text(0.35, -0.2, 'Training Data', color=colors['text'], fontsize=8,
                     horizontalalignment='center')
    sample_plot_ax.text(0.85, -0.2, 'Test', color=colors['text'], fontsize=8,
                     horizontalalignment='center')
    
    # --------------------------
    # 4. Bottom Left: Feature Engineering
    # --------------------------
    feature_eng_ax = plt.subplot(gs[2, 0:2])
    feature_eng_ax.set_facecolor(colors['background'])
    
    # Title for feature engineering section
    feature_eng_ax.text(0.5, 0.95, 'Feature Engineering', 
                    fontsize=16, fontweight='bold', color=colors['text'],
                    horizontalalignment='center')
    
    # Create feature engineering steps
    features = [
        {'name': 'Technical Indicators', 'desc': '• Moving Averages (10, 30, 60 day)\n• Volatility Measures\n• Relative Price to MA\n• RSI (Relative Strength Index)'},
        {'name': 'Relative Features', 'desc': '• Stock vs Bond Performance\n• Asset Relative Momentum\n• Price Change Acceleration'},
        {'name': 'Risk Metrics', 'desc': '• Rolling Window Volatility\n• Drawdown Calculations\n• Risk-Free Rate Changes'},
        {'name': 'Data Normalization', 'desc': '• RobustScaler Applied\n• Outlier Handling\n• Feature Clipping (-3 to 3)'}
    ]
    
    # Place feature engineering steps in boxes
    y_positions = np.linspace(0.8, 0.15, len(features))
    for i, feature in enumerate(features):
        feature_rect = patches.Rectangle((0.05, y_positions[i]-0.08), 0.4, 0.16, 
                                      linewidth=2, edgecolor=colors['feature_eng'], 
                                      facecolor=colors['feature_eng'], alpha=0.2)
        feature_eng_ax.add_patch(feature_rect)
        feature_eng_ax.text(0.25, y_positions[i]+0.04, feature['name'], 
                        fontsize=12, color=colors['text'], 
                        horizontalalignment='center', verticalalignment='center')
        feature_eng_ax.text(0.25, y_positions[i]-0.02, feature['desc'], 
                        fontsize=8, color=colors['text'], 
                        horizontalalignment='center', verticalalignment='center')
    
    # Create code example for feature engineering
    code_box = patches.Rectangle((0.55, 0.15), 0.4, 0.65, 
                             linewidth=1, edgecolor=colors['border'], 
                             facecolor='white', alpha=0.8)
    feature_eng_ax.add_patch(code_box)
    feature_eng_ax.text(0.75, 0.75, 'Feature Engineering Code', 
                     fontsize=10, color=colors['text'], 
                     horizontalalignment='center', verticalalignment='center')
    
    code_snippet = """
# Moving averages
df["SP500_MA10"] = df["SP500"].rolling(
    window=10).mean()
df["SP500_MA30"] = df["SP500"].rolling(
    window=30).mean()

# Volatility metrics
df["SP500_Vol20"] = df["SP500_Return"].rolling(
    window=20).std()

# Relative to moving average
df["SP500_RelToMA"] = (
    df["SP500"] / df["SP500_MA30"]) - 1

# Relative Strength Index (RSI)
df["SP500_RSI"] = calculate_rsi(df["SP500"], 14)

# Relative performance
df["SP500_vs_Bond_Return"] = (
    df["SP500_Return"] - df["Bond_Return"])

# Normalize features
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[feature_columns] = scaler.fit_transform(
    df[feature_columns])

# Clip extreme values
df[feature_columns] = df[feature_columns].clip(-3, 3)
    """
    feature_eng_ax.text(0.75, 0.45, code_snippet, 
                     fontsize=6, color=colors['text'], 
                     horizontalalignment='center', verticalalignment='center',
                     family='monospace')
    
    # Remove axes
    feature_eng_ax.set_xticks([])
    feature_eng_ax.set_yticks([])
    feature_eng_ax.spines['top'].set_visible(False)
    feature_eng_ax.spines['right'].set_visible(False)
    feature_eng_ax.spines['bottom'].set_visible(False)
    feature_eng_ax.spines['left'].set_visible(False)
    
    # --------------------------
    # 5. Bottom Right: Data Splitting
    # --------------------------
    split_ax = plt.subplot(gs[2, 2])
    split_ax.set_facecolor(colors['background'])
    
    # Title for data splitting section
    split_ax.text(0.5, 0.95, 'Data Splitting & Final Format', 
              fontsize=16, fontweight='bold', color=colors['text'],
              horizontalalignment='center')
    
    # Create data splitting visualization
    split_rect = patches.Rectangle((0.1, 0.65), 0.8, 0.2, 
                                linewidth=2, edgecolor=colors['model_data'], 
                                facecolor=colors['model_data'], alpha=0.2)
    split_ax.add_patch(split_rect)
    split_ax.text(0.5, 0.75, 'Chronological Split at 2020-01-01', 
              fontsize=12, color=colors['text'], 
              horizontalalignment='center')
    
    # Add arrows for training and testing splits
    arrow_props = dict(arrowstyle="->", connectionstyle="arc3,rad=0", color=colors['arrow'], lw=1.5)
    
    split_ax.annotate("Training Data", xy=(0.3, 0.5), xytext=(0.3, 0.65),
                   arrowprops=arrow_props, fontsize=10, color=colors['text'],
                   horizontalalignment='center')
    
    split_ax.annotate("Testing Data", xy=(0.7, 0.5), xytext=(0.7, 0.65),
                   arrowprops=arrow_props, fontsize=10, color=colors['text'],
                   horizontalalignment='center')
    
    # Create final format tables (simplified)
    train_rect = patches.Rectangle((0.1, 0.15), 0.35, 0.35, 
                                linewidth=2, edgecolor=colors['model_data'], 
                                facecolor=colors['model_data'], alpha=0.2)
    split_ax.add_patch(train_rect)
    split_ax.text(0.27, 0.42, 'Training Data', 
              fontsize=10, color=colors['text'], 
              horizontalalignment='center')
    split_ax.text(0.27, 0.35, '1960-01-01 to 2019-12-31', 
              fontsize=8, color=colors['text'], 
              horizontalalignment='center')
    split_ax.text(0.27, 0.28, f'22 Features x 15,000 Days', 
              fontsize=8, color=colors['text'], 
              horizontalalignment='center')
    split_ax.text(0.27, 0.22, 'Normalized, No NaNs', 
              fontsize=8, color=colors['text'], 
              horizontalalignment='center')
    
    test_rect = patches.Rectangle((0.55, 0.15), 0.35, 0.35, 
                               linewidth=2, edgecolor=colors['model_data'], 
                               facecolor=colors['model_data'], alpha=0.2)
    split_ax.add_patch(test_rect)
    split_ax.text(0.72, 0.42, 'Testing Data', 
              fontsize=10, color=colors['text'], 
              horizontalalignment='center')
    split_ax.text(0.72, 0.35, '2020-01-01 to 2023-12-31', 
              fontsize=8, color=colors['text'], 
              horizontalalignment='center')
    split_ax.text(0.72, 0.28, f'22 Features x 1,000 Days', 
              fontsize=8, color=colors['text'], 
              horizontalalignment='center')
    split_ax.text(0.72, 0.22, 'Same Scaling as Training', 
              fontsize=8, color=colors['text'], 
              horizontalalignment='center')
    
    # Remove axes
    split_ax.set_xticks([])
    split_ax.set_yticks([])
    split_ax.spines['top'].set_visible(False)
    split_ax.spines['right'].set_visible(False)
    split_ax.spines['bottom'].set_visible(False)
    split_ax.spines['left'].set_visible(False)
    
    # --------------------------
    # 6. Bottom: Data Flow to Models
    # --------------------------
    models_ax = plt.subplot(gs[3, :])
    models_ax.set_facecolor(colors['background'])
    
    # Title for models section
    models_ax.text(0.5, 0.8, 'Data Flow to Models', 
                fontsize=16, fontweight='bold', color=colors['text'],
                horizontalalignment='center')
    
    # Create model boxes
    mvo_rect = patches.Rectangle((0.1, 0.1), 0.35, 0.5, 
                              linewidth=2, edgecolor=colors['mvo'], 
                              facecolor=colors['mvo'], alpha=0.2)
    models_ax.add_patch(mvo_rect)
    models_ax.text(0.27, 0.45, 'Mean-Variance Optimization', 
                fontsize=12, color=colors['text'], 
                horizontalalignment='center')
    models_ax.text(0.27, 0.35, 'Inputs:', 
                fontsize=10, color=colors['text'], 
                horizontalalignment='center')
    models_ax.text(0.27, 0.25, '• Expected Returns\n• Covariance Matrix\n• Risk Aversion Parameter', 
                fontsize=8, color=colors['text'], 
                horizontalalignment='center')
    
    rl_rect = patches.Rectangle((0.55, 0.1), 0.35, 0.5, 
                             linewidth=2, edgecolor=colors['deep_rl'], 
                             facecolor=colors['deep_rl'], alpha=0.2)
    models_ax.add_patch(rl_rect)
    models_ax.text(0.72, 0.45, 'Deep Reinforcement Learning', 
                fontsize=12, color=colors['text'], 
                horizontalalignment='center')
    models_ax.text(0.72, 0.35, 'Environment State:', 
                fontsize=10, color=colors['text'], 
                horizontalalignment='center')
    models_ax.text(0.72, 0.25, '• All Feature Columns\n• Historical Price/Return Data\n• Custom PortfolioAllocationEnv', 
                fontsize=8, color=colors['text'], 
                horizontalalignment='center')
    
    # Add arrows for data flow
    # From preprocessing to split
    draw_data_flow_arrow(preprocessing_ax, (0.5, 0.05), feature_eng_ax, (0.5, 0.9))
    
    # From feature eng to split
    draw_data_flow_arrow(feature_eng_ax, (0.5, 0.05), models_ax, (0.5, 0.9))
    
    # From split to models
    draw_data_flow_arrow(split_ax, (0.3, 0.05), models_ax, (0.27, 0.65))
    draw_data_flow_arrow(split_ax, (0.7, 0.05), models_ax, (0.72, 0.65))
    
    # From data sources to preprocessing
    draw_data_flow_arrow(data_sources_ax, (0.2, 0.1), preprocessing_ax, (0.25, 0.9))
    draw_data_flow_arrow(data_sources_ax, (0.5, 0.1), preprocessing_ax, (0.5, 0.9))
    draw_data_flow_arrow(data_sources_ax, (0.8, 0.1), preprocessing_ax, (0.75, 0.9))
    
    # Remove axes
    models_ax.set_xticks([])
    models_ax.set_yticks([])
    models_ax.spines['top'].set_visible(False)
    models_ax.spines['right'].set_visible(False)
    models_ax.spines['bottom'].set_visible(False)
    models_ax.spines['left'].set_visible(False)
    
    # Add title
    plt.suptitle('Data Sources and Preprocessing Pipeline for Portfolio Allocation System', 
                fontsize=24, fontweight='bold', color=colors['text'], y=0.98)
    
    # Add caption
    plt.figtext(0.5, 0.01, 
               'The portfolio allocation system processes financial market data, macroeconomic indicators, and synthetic allocations\nthrough a multi-stage pipeline for feature engineering and model training.',
               fontsize=10, color=colors['text'], 
               horizontalalignment='center')
    
    # Final adjustments
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Data preprocessing visualization saved to {save_path}")

def draw_data_flow_arrow(ax1, start, ax2, end, color='#95a5a6'):
    """
    Draw an arrow from ax1 to ax2 with coordinates in respective axes.
    """
    # Convert to figure coordinates
    fig = plt.gcf()
    start_fig = ax1.transData.transform(start)
    start_fig = fig.transFigure.inverted().transform(start_fig)
    
    end_fig = ax2.transData.transform(end)
    end_fig = fig.transFigure.inverted().transform(end_fig)
    
    # Draw the arrow
    arrow_props = dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", 
                      color=color, lw=1.5)
    
    plt.annotate("", 
                xy=end_fig, xycoords='figure fraction',
                xytext=start_fig, textcoords='figure fraction',
                arrowprops=arrow_props)

if __name__ == "__main__":
    create_data_preprocessing_visualization("data_preprocessing_pipeline.png")