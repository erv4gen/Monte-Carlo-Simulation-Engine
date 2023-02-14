from locale import currency
import sys , os


sys.path.append( os.path.abspath(os.path.curdir))
import unittest
import numpy as np

import mc.executor as simulator
import mc.analysis as analysis
from mc.executor import initialize_executors
from mc.utils import StrategyParams
from mc.assets import *

class TestEuropeanNaiveCall(unittest.TestCase):

    def test_call_option(self):
        premium = 0.05
        call_option = EuropeanNaiveCallOption(premium)
        amount = 1
        S0 = 100
        K = 110
        T1 = 60
        premium_payed = call_option.write(S0,K,amount,T1)

        self.assertTrue(np.isclose(premium_payed, S0 *premium  ))

        self.assertFalse(call_option.decay(31))

        self.assertTrue(call_option.decay(60))

        ITM_assigment = call_option.assign(110)


        self.assertTrue(ITM_assigment.amount >0.)
        self.assertTrue(ITM_assigment.current_price== K)

        second_assigmne_should_be_none = call_option.assign(110)

        self.assertTrue(second_assigmne_should_be_none.value==0.0)
        




    def test_put_option(self):
        premium = 0.05
        put_option = EuropeanNaivePutOption(premium)
        amount = 1
        S0 = 100
        K = 90
        T1 = 60
        premium_payed = put_option.write(S0,K,amount,T1)

        self.assertTrue(np.isclose(premium_payed, S0 *premium  ))

        self.assertFalse(put_option.decay(31))

        self.assertTrue(put_option.decay(60))

        ITM_assigment = put_option.assign(90)


        self.assertTrue(ITM_assigment.amount >0.)
        self.assertTrue(ITM_assigment.current_price== K)

        second_assigmne_should_be_none = put_option.assign(90)

        self.assertTrue(second_assigmne_should_be_none.value==0.0)

class TestReturnsCalculator(unittest.TestCase):
    def test_calculate_returns(self):
        # Create a test case with arbitrary values for allocated_capital
        allocated_capital = np.array([[[1, 2], [3, 4], [5, 6]]
                                    , [[7, 8], [9, 10], [11, 12]]])
        
        # Create an instance of the ReturnsCalculator class
        calculator = analysis.ReturnsCalculator(allocated_capital)
        
        # Run the calculate_returns method
        calculator.calculate_returns()
        
        # Assert that sim_portfolio is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_portfolio, np.array([[3, 7, 11], [15, 19, 23]])))
        
        # Assert that sim_retuns is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_retuns, np.array([[0.,1.33333333, 0.57142857], [0.,0.26666667, 0.21052632]])))
        
        # Assert that sim_cum_retuns is calculated correctly
        self.assertTrue(np.allclose(calculator.sim_cum_retuns, np.array([[1., 2.33333333, 3.66666667],[1., 1.26666667, 1.53333333] ])))



class TestPortfolioClass(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.initial_price = 100.0
        self.def_params = StrategyParams()
        self.split_params = StrategyParams(percent_allocated=0.5)
        self.asset_ticker = Symbols.ETH
    def test_portfolio_init(self):
        sim_portfolio = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.def_params)[0]
        self.assertTrue(np.allclose(sim_portfolio.capital,self.initial_price))

    def test_buy_equity_not_enough(self):
        
        portfolio = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.def_params)[0]
        initial_price = portfolio.equity.initial_price
        buy_price = initial_price + 1
        buy_amount = 10
        portfolio.log_asset_price(buy_price)
        
        with self.assertRaises(NotEnoughMoney):
            portfolio.buy_equity(buy_amount)
        
    def test_buy_equity_enough(self):
        

        trader = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.split_params)[0]
        buy_price = self.initial_price + 1
        trader.log_asset_price(buy_price)
        buy_amount = 0.25
        cost = buy_amount * buy_price
        initial_amount = trader.equity.amount
        initial_cash_amount = trader.cash.amount
        trader.buy_equity(buy_amount)

        self.assertEqual(trader.equity.amount, initial_amount + buy_amount)
        self.assertAlmostEqual(trader.equity.initial_price, (initial_amount * self.initial_price + buy_amount * buy_price) / (initial_amount + buy_amount))
        self.assertAlmostEqual(trader.cash.amount, initial_cash_amount-cost)
        
    def test_sell_equity(self):
        trader = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.split_params)[0]

        initial_equity_amount = trader.equity.amount
        initial_cash_amount = trader.cash.amount
        sell_price = trader.portfolio.equity.get_asset(self.asset_ticker).current_price + 1

        trader.log_asset_price(sell_price)
        sell_amount = 0.25
        
        trader.sell_equity(sell_amount)
        
        self.assertEqual(trader.equity.amount, initial_equity_amount - sell_amount)
        self.assertAlmostEqual(trader.cash.amount, initial_cash_amount + sell_amount * sell_price)
        

    def test_sell_equity_not_enough(self):
        portfolio = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.split_params)[0]
        

        sell_price = portfolio._equity.initial_price + 1

        portfolio.log_asset_price(sell_price)
        sell_amount = 1.25
        with self.assertRaises(NotEnoughAmount):
            portfolio.sell_equity(sell_amount)
        

    def test_rebalancer(self):
        portfolio = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.split_params)[0]
        target_share = 0.5
        price_changed = 30
        portfolio.log_asset_price(price_changed)
        current_shares = portfolio.portfolio_balance
        self.assertTrue(current_shares.cash > current_shares.equity)
        portfolio.rebalance(target_share=target_share)

        new_share = portfolio.portfolio_balance
        self.assertTrue( np.isclose(new_share.cash, new_share.equity))
class TestExecutorClass(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.initial_price = 1.0
        self.split_params = StrategyParams(percent_allocated=0.5,max_rebalances=100,rebalance_threshold_down=0.5,rebalance_threshold_up=1.5)

        self.options_params = StrategyParams(percent_allocated=0.5,max_rebalances=100,rebalance_threshold_down=0.5,rebalance_threshold_up=1.5
                            ,option_duration=2,option_every_itervals=2,option_premium=0.1)
        

        self.time_series = np.array([[1.,1.25,1.05,1.15,1.30,1.35]])
        self.expected_portfolio_5050_no_rebalance = np.array([[1.0, 1.125, 1.025, 1.075, 1.15, 1.175]])

        self.time_series_sudden_drop = np.array([[1.,1.10,0.49]])
        self.expected_portfolio_5050_sudden_drop_rebalance_50pct = np.array([[1.
                                                                            ,1.05 # 0.5 + 0.55 = 1.05 #first increment
                                                                            ,0.745     #0.5 + 0.5 * 0.445454 = 0.725 #portfolio  value
                                                                                        #change 0.755/2 = 0.3525
                                                                                        
                                                                            ,
                                                                            ]])

        self.time_series_sudden_up = np.array([[1.,1.10,4.49]])
        self.expected_portfolio_5050_sudden_up_rebalance_150pct = np.array([[1.,1.05,2.745]])

    def test_price_tracker_5050_no_rebalance(self):
        portfolios = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.split_params)
        sim_tracker = simulator.SimulationTracker(self.time_series,portfolios,self.split_params)
        sim_tracker.run_simulations()

        # Create an instance of the ReturnsCalculator class
        calculator = simulator.ReturnsCalculator(sim_tracker.allocated_capital)
        
        # Run the calculate_returns method
        calculator.calculate_returns()

        
        self.assertTrue(np.allclose(calculator.sim_portfolio, self.expected_portfolio_5050_no_rebalance))
        
    def test_price_tracker_5050_rebalance_down(self):
        portfolios = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.split_params)
        sim_tracker = (simulator
                        .SimulationTracker(self.time_series_sudden_drop,portfolios,self.split_params)
                        .add_rebalance_below(0.5)
                        .run_simulations()
                        )

        calculator = (simulator.ReturnsCalculator(sim_tracker.allocated_capital)
                        .calculate_returns()
                        )
        self.assertTrue(np.allclose(calculator.sim_portfolio, self.expected_portfolio_5050_sudden_drop_rebalance_50pct))

    def test_price_tracker_5050_rebalance_up(self):
        portfolios = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.split_params)
        sim_tracker = (simulator
                        .SimulationTracker(self.time_series_sudden_up,portfolios,self.split_params)
                        .add_rebalance_above(1.5)
                        .run_simulations()
                        )

        calculator = (simulator.ReturnsCalculator(sim_tracker.allocated_capital)
                        .calculate_returns()
                        )
        self.assertTrue(np.allclose(calculator.sim_portfolio, self.expected_portfolio_5050_sudden_up_rebalance_150pct))

    def test_call_option_write(self):
        portfolios = initialize_executors(n=1,initial_price=self.initial_price,strategy_params=self.options_params)
        sim_tracker = (simulator
                        .SimulationTracker(self.time_series,portfolios,self.options_params)
                        .run_simulations()
                        )
        pass
class TestDecigionLogic(unittest.TestCase):
    def test_threshold_below(self):
        
        self.assertFalse(simulator.check_is_below_threshold(current_price=1,prev_price=0.5,threshold=0.5))
        self.assertTrue(simulator.check_is_below_threshold(current_price=0.49,prev_price=1,threshold=0.5))

    def test_threshold_above(self):
        
        self.assertFalse(simulator.check_is_above_threshold(current_price=0.49,prev_price=1,threshold=0.5))
        self.assertTrue(simulator.check_is_above_threshold(current_price=1,prev_price=0.5,threshold=0.5))

if __name__ == '__main__':
    unittest.main()