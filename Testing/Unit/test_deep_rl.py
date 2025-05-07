import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model functions
from Models.MVO import optimize_portfolio, calculate_statistics

class TestMVO(unittest.TestCase):
    
    def setUp(self):
        # Create simple test data
        self.expected_returns = np.array([0.05, 0.03])  # 5% stocks, 3% bonds
        self.cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])  # Covariance matrix
        self.risk_aversion = 2.0
        
    def test_optimize_portfolio(self):
        # Test basic optimization functionality
        weights = optimize_portfolio(self.expected_returns, self.cov_matrix, self.risk_aversion)
        
        # Check if weights sum to 1 (ensure proper allocation)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        
        # Check if weights are within bounds [0,1]
        self.assertTrue(all(w >= 0 and w <= 1 for w in weights))
        
        # Higher risk aversion should allocate less to risky asset (stocks)
        conservative_weights = optimize_portfolio(self.expected_returns, self.cov_matrix, 5.0)
        self.assertLess(conservative_weights[0], weights[0])  # Less allocation to stocks
        
    def test_calculate_statistics(self):
        # Create a simple DataFrame to test statistics calculation
        dates = pd.date_range(start='2020-01-01', periods=100)
        data = pd.DataFrame({
            'Date': dates,
            'SP500_Return': np.random.normal(0.0005, 0.01, 100),  # ~12% annual return, ~16% volatility
            'USBIG_Return': np.random.normal(0.0002, 0.005, 100)  # ~5% annual return, ~8% volatility
        })
        
        # Calculate statistics
        returns, cov_matrix = calculate_statistics(data)
        
        # Check dimensions
        self.assertEqual(len(returns), 2)
        self.assertEqual(cov_matrix.shape, (2, 2))
        
        # Check positive definiteness of covariance matrix
        eigenvalues = np.linalg.eigvals(cov_matrix)
        self.assertTrue(all(eigenvalues > 0))

if __name__ == '__main__':
    unittest.main()