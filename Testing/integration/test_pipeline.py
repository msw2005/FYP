import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import necessary components
try:
    from Models.MVO import optimize_portfolio, calculate_statistics
    from Backend.Models.models import load_and_preprocess_data
except ImportError:
    try:
        from Models.models import load_and_preprocess_data
    except ImportError:
        print("Warning: Could not import pipeline components. Tests will be skipped.")

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        # Skip tests if modules couldn't be imported
        if 'load_and_preprocess_data' not in globals() or 'optimize_portfolio' not in globals():
            self.skipTest("Required modules not available")
        
        # Define paths to data files for testing
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, "Data", "Processed")
        
        # Check if expected data files exist
        self.sp500_file = os.path.join(self.data_dir, "processed_sp500_data.csv")
        self.bond_file = os.path.join(self.data_dir, "processed_bond_data.csv")
        self.rf_file = os.path.join(self.data_dir, "processed_risk_free_rate_data.csv")
        
        self.data_available = all(os.path.exists(f) for f in [self.sp500_file, self.bond_file, self.rf_file])
    
    def test_data_loading(self):
        """Test that data can be loaded and preprocessed"""
        if not self.data_available:
            self.skipTest("Required data files not available")
        
        try:
            # Load data with default split date
            train_data, test_data, risk_aversion, scaler = load_and_preprocess_data()
            
            # Check that we got valid dataframes
            self.assertIsInstance(train_data, pd.DataFrame)
            self.assertIsInstance(test_data, pd.DataFrame)
            
            # Check that we have expected columns
            required_columns = ['SP500_Return', 'Bond_Return', 'RiskFreeRate']
            for col in required_columns:
                self.assertIn(col, train_data.columns)
                self.assertIn(col, test_data.columns)
            
            # Check that data is properly split
            if 'Date' in train_data.columns and 'Date' in test_data.columns:
                train_max_date = pd.to_datetime(train_data['Date']).max()
                test_min_date = pd.to_datetime(test_data['Date']).min()
                self.assertLess(train_max_date, test_min_date)
        
        except Exception as e:
            self.fail(f"Data loading failed with error: {e}")
    
    def test_end_to_end_optimization(self):
        """Test the full optimization pipeline from data to weights"""
        if not self.data_available:
            self.skipTest("Required data files not available")
        
        try:
            # 1. Load data
            train_data, test_data, risk_aversion, scaler = load_and_preprocess_data()
            
            # 2. Calculate statistics on train data
            train_returns, train_cov_matrix = calculate_statistics(train_data)
            
            # 3. Optimize portfolio
            weights = optimize_portfolio(train_returns, train_cov_matrix, risk_aversion)
            
            # 4. Verify results
            self.assertEqual(len(weights), 2)  # Should have 2 weights (stocks, bonds)
            self.assertAlmostEqual(sum(weights), 1.0)  # Weights should sum to 1
            self.assertTrue(all(w >= 0 and w <= 1 for w in weights))  # Weights should be in [0,1]
            
            # 5. Calculate test statistics for validation
            test_returns, test_cov_matrix = calculate_statistics(test_data)
            test_weights = optimize_portfolio(test_returns, test_cov_matrix, risk_aversion)
            
            # 6. Verify test results
            self.assertEqual(len(test_weights), 2)  # Should have 2 weights (stocks, bonds)
            self.assertAlmostEqual(sum(test_weights), 1.0)  # Weights should sum to 1
            self.assertTrue(all(w >= 0 and w <= 1 for w in test_weights))  # Weights should be in [0,1]
        
        except Exception as e:
            self.fail(f"End-to-end optimization failed with error: {e}")

if __name__ == '__main__':
    unittest.main()