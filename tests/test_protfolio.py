import sys , os
sys.path.append( os.path.abspath(os.path.curdir))
import unittest
import numpy as np
import mc.portfolio as simulator
from mc.portfolio import initialize_portfolios
from mc.utils import StrategyParams
class TestReturnsCalculator(unittest.TestCase):
    def test_calculate_returns(self):
        # Create a test case with arbitrary values for allocated_capital
        allocated_capital = np.array([[[1, 2], [3, 4], [5, 6]]
                                    , [[7, 8], [9, 10], [11, 12]]])
        
        # Create an instance of the ReturnsCalculator class
        calculator = simulator.ReturnsCalculator(allocated_capital)
        
        # Run the calculate_returns method
        calculator.calculate_returns()
        
        # Assert that sim_portfolio is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_portfolio, np.array([[3, 7, 11], [15, 19, 23]])))
        
        # Assert that sim_retuns is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_retuns, np.array([[0.,1.33333333, 0.57142857], [0.,0.26666667, 0.21052632]])))
        
        # Assert that sim_cum_retuns is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_cum_retuns, np.array([[1., 2.33333333, 3.66666667],[1., 1.26666667, 1.53333333] ])))



class TestPortfolioClass(unittest.TestCase):
    def test_portfolio_init(self):
        params = StrategyParams()
        initial_price = 100.0
        sim_portfolio = initialize_portfolios(n=1,initial_price=initial_price,strategy_params=params)[0]

        self.assertTrue(np.allclose(sim_portfolio.capital,initial_price))


if __name__ == '__main__':
    unittest.main()