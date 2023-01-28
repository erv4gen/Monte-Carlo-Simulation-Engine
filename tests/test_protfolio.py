import sys , os
sys.path.append( os.path.abspath(os.path.curdir))
import unittest
import numpy as np
from mc.portfolio import ReturnsCalculator
class TestReturnsCalculator(unittest.TestCase):
    def test_calculate_returns(self):
        # Create a test case with arbitrary values for allocated_capital
        allocated_capital = np.array([[[1, 2], [3, 4], [5, 6]]
                                    , [[7, 8], [9, 10], [11, 12]]])
        
        # Create an instance of the ReturnsCalculator class
        calculator = ReturnsCalculator(allocated_capital)
        
        # Run the calculate_returns method
        calculator.calculate_returns()
        
        # Assert that sim_portfolio is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_portfolio, np.array([[3, 7, 11], [15, 19, 23]])))
        
        # Assert that sim_retuns is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_retuns, np.array([[0.,1.33333333, 0.57142857], [0.,0.26666667, 0.21052632]])))
        
        # Assert that sim_cum_retuns is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_cum_retuns, np.array([[1., 2.33333333, 3.66666667],[1., 1.26666667, 1.53333333] ])))

if __name__ == '__main__':
    unittest.main()