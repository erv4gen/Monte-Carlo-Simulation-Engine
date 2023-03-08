import sys , os


sys.path.append( os.path.abspath(os.path.curdir))
import unittest
import numpy as np

import mc.executor as simulator
import mc.analysis as analysis
from mc.executor import initialize_executors
from mc.utils import StrategyParams , Env
from mc.assets import *
from mc.pricing import *
from mc import constants

env = Env().create_test_env()


class TestPricingClass(unittest.TestCase):
    def test_base_function(self):
        spot_price =100.
        strike = 110.
        maturity = 31
        volatility =0.25
        dividend_rate = 0.0
        option_type = OptionType.CALL
        risk_free_rate= 0.04

        init_date = ql.Date.todaysDate()
        eval_date = init_date+ ql.Period(10, ql.Days)

        ql.Settings.instance().evaluationDate = init_date
        option = create_option(spot_price,strike, maturity, volatility, risk_free_rate, option_type,dividend_rate)
        initial_price = option.NPV()

        ql.Settings.instance().evaluationDate = eval_date

        eval_price = option.NPV()
        
        self.assertLess(eval_price,initial_price)

    def test_option_pricing_price_change(self):
        spot_price =100
        strike = 110
        maturity = 31
        volatility =0.25
        dividend_rate = 0.0
        option_type = OptionType.CALL
        risk_free_rate= 0.04
        call = QlEuropeanOption(spot_price,strike, maturity, volatility, risk_free_rate,dividend_rate, option_type)
        initial_price = call.NPV()

        days_passed = 10
        new_price = 130
        
        price_after_t = call.decay(days_passed).price_drift(new_price).NPV()
        
        self.assertTrue(initial_price<price_after_t)

    # def test_option_pricing_price_not_change(self):
    #     spot_price =110
    #     strike = 100
    #     maturity = 31
    #     volatility =0.25
    #     dividend_rate = 0.0
    #     option_type = OptionType.CALL
    #     risk_free_rate= 0.04
    #     call = QlEuropeanOption(spot_price,strike, maturity, volatility, risk_free_rate,dividend_rate, option_type)
    #     initial_price = call.NPV()
    #     price_trace = []
    #     for days_passed in range(1,10):
    #         npv_ =call.decay(days_passed).NPV()
    #         ql.Settings.instance().evaluationDate = call._eval_date
    #         npv_ =call.NPV()
    #         price_trace.append(npv_)

    #     self.assertTrue(price_trace[0]> price_trace[-1])

class TestAssetClass(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.def_params = StrategyParams()
        self.initial_price = 100.0
        self.asset_ticker = Symbols.ETH
        self.interest = 0.05
        self.rate_ = (1+self.interest/constants.AnnualTimeInterval.days.value)
        self.ts_params = dict(current_price=self.initial_price,sigma=0.3)
                
    def test_capitalization_cash(self):
        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.def_params)[0]
        cash =  trader.portfolio.cash 
        initial_amount = cash.amount
        

        cash.capitalize(self.rate_)

        self.assertAlmostEqual(cash.amount , initial_amount * self.rate_)


    def test_capitalization_asset(self):
        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.def_params)[0]
        asset = trader.portfolio.equity.get_asset(self.asset_ticker)
        initial_amount = asset.amount

        asset.capitalize(self.rate_)
        self.assertAlmostEqual(asset.amount , initial_amount * self.rate_)

class TestEuropeanNaiveCall(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ticker= Symbols.ETH
        self.initial_price = 100
        self.split_params = StrategyParams(percent_allocated=0.5)
        self.ts_params = dict(current_price=self.initial_price,sigma=0.3)
    def test_call_option(self):
        volatility = 0.25
        risk_free_rate = 0.05
        call_option = EuropeanNaiveCallOption(ticker=self.ticker,volatility=volatility,risk_free_rate=risk_free_rate)
        amount = 1
        S0 = 100
        K = 110
        T1 = 60
        call_option.write(S0,K,amount,T1)

        self.assertLess(call_option.premium, S0 *risk_free_rate  )

        self.assertFalse(call_option.decay(31))

        self.assertTrue(call_option.decay(60))

        ITM_assigment = call_option.assign(110)


        self.assertTrue(ITM_assigment.amount >0.)
        self.assertTrue(ITM_assigment.current_price== K)

        second_assigmne_should_be_none = call_option.assign(110)

        self.assertTrue(second_assigmne_should_be_none.value==0.0)
        




    def test_put_option(self):
        premium = 0.05
        risk_free_rate = 0.05
        put_option = EuropeanNaivePutOption(ticker=self.ticker,volatility= premium,risk_free_rate=risk_free_rate)
        amount = 1
        S0 = 100
        K = 90
        T1 = 60
        put_option.write(S0,K,amount,T1)

        self.assertLess(put_option.premium, S0 *risk_free_rate  )

        self.assertFalse(put_option.decay(31))

        self.assertTrue(put_option.decay(60))

        ITM_assigment = put_option.assign(90)


        self.assertTrue(ITM_assigment.amount >0.)
        self.assertTrue(ITM_assigment.current_price== K)

        second_assigmne_should_be_none = put_option.assign(90)

        self.assertTrue(second_assigmne_should_be_none.value==0.0)

    def test_assigment_put_ITM(self):
        risk_free_rate = 0.05
        volatility = 0.35
        amount = 0.25
        S0 = 100
        K = 90
        T1 = 60

        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)[0]
        asset = trader.portfolio.equity.get_asset(self.ticker)


        initial_equity_amount = asset.amount
        initial_cash_amount = trader.portfolio.cash.amount
        initial_equlity_value = asset.value
        put_option = (EuropeanNaivePutOption(ticker=self.ticker,volatility= volatility,risk_free_rate=risk_free_rate)
                        .write(S0,K,amount,T1)
                     )
        
        self.assertGreater(put_option.premium +initial_cash_amount, initial_cash_amount)

        S1 = 80

        asset_delivery = put_option.assign(S1)

        trader.sell_equity(asset,asset_delivery.amount,asset_delivery.current_price)

        new_value = asset.value

        self.assertTrue(new_value< initial_equlity_value)

    def test_assigment_call_ITM(self):
        volatility = 0.35
        risk_free_rate= 0.05
        amount = 0.25
        S0 = 100
        K = 110
        T1 = 60

        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)[0]
        asset = trader.portfolio.equity.get_asset(self.ticker)


        initial_equity_amount = asset.amount
        initial_cash_amount = trader.portfolio.cash.amount
        initial_equlity_value = asset.value
        call_option = (EuropeanNaiveCallOption(ticker=self.ticker,volatility= volatility,risk_free_rate=risk_free_rate)
                        .write(S0,K,amount,T1)
                     )
        
        self.assertGreater(call_option.premium +initial_cash_amount,initial_cash_amount)

        S1 = 120

        asset_delivery = call_option.assign(S1)

        trader.buy_equity(asset,asset_delivery.amount,asset_delivery.current_price)

        new_value = asset.value

        self.assertTrue(asset.initial_price > S0)


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
        self.ts_params = dict(current_price=self.initial_price,sigma=0.3)

        

    def test_portfolio_init(self):
        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.def_params)[0]
        asset = trader.portfolio.equity.get_asset(self.asset_ticker)
        self.assertTrue(np.allclose(asset.initial_price ,self.initial_price))

    def test_buy_equity_not_enough(self):
        
        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.def_params)[0]
        asset = trader.portfolio.equity.get_asset(self.asset_ticker)
        initial_price = asset.initial_price
        buy_price = initial_price + 1
        buy_amount = 10

        
        trader.portfolio.equity.log_asset_price(asset,buy_price)
        
        with self.assertRaises(NotEnoughMoney):
            trader.buy_equity(asset,buy_amount)
        
    def test_buy_equity_enough(self):
        

        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)[0]
        buy_price = self.initial_price - 1

        asset = trader.portfolio.equity.get_asset(self.asset_ticker)
        trader.portfolio.equity.log_asset_price(asset,buy_price)
        buy_amount = 0.25
        cost = buy_amount * buy_price
        initial_amount = asset.amount
        initial_cash_amount = trader.portfolio.cash.amount
        trader.buy_equity(asset,buy_amount)

        self.assertEqual(asset.amount, initial_amount + buy_amount)
        self.assertAlmostEqual(asset.initial_price, (initial_amount * self.initial_price + buy_amount * buy_price) / (initial_amount + buy_amount))
        self.assertAlmostEqual(trader.portfolio.cash.amount, initial_cash_amount-cost)
        
    def test_sell_equity(self):
        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)[0]
        asset = trader.portfolio.equity.get_asset(self.asset_ticker)


        initial_equity_amount = asset.amount
        initial_cash_amount = trader.portfolio.cash.amount
        sell_price = asset.current_price + 1

        
        trader.portfolio.equity.log_asset_price(asset,sell_price)
        sell_amount = 0.25
        
        trader.sell_equity(asset,sell_amount)
        
        self.assertEqual(asset.amount, initial_equity_amount - sell_amount)
        self.assertAlmostEqual(trader.portfolio.cash.amount, initial_cash_amount + sell_amount * sell_price)
        

    def test_sell_equity_not_enough(self):
        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)[0]
        asset = trader.portfolio.equity.get_asset(self.asset_ticker)

        
        sell_price = trader.portfolio.equity.get_asset(self.asset_ticker).current_price + 1

        trader.portfolio.equity.log_asset_price(asset,sell_price)
        sell_amount = 1.25
        with self.assertRaises(NotEnoughAmount):
            trader.sell_equity(asset,sell_amount)
        

    def test_rebalancer(self):
        trader = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)[0]
        asset = trader.portfolio.equity.get_asset(self.asset_ticker)
        target_share = 0.5
        new_price = 30
        trader.portfolio.equity.log_asset_price(asset,new_price)
        
        current_shares = trader.portfolio.share_balance
        self.assertTrue(current_shares.cash > current_shares.equity)
        trader.rebalance(asset,target_share=target_share)

        new_share = trader.portfolio.share_balance
        self.assertTrue( np.isclose(new_share.cash, new_share.equity))
class TestExecutorClass(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.initial_price = 1.0
        self.ts_params = dict(current_price=self.initial_price,sigma=0.3)
        self.split_params = StrategyParams(percent_allocated=0.5,max_rebalances=100,rebalance_threshold_down=0.5,rebalance_threshold_up=1.5)

        self.options_params = StrategyParams(percent_allocated=0.5,max_rebalances=100,rebalance_threshold_down=0.5,rebalance_threshold_up=1.5
                            ,option_duration=2,option_every_itervals=3)
        

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


        self.time_series_sudden_drop_and_bounce_back = np.array([[1.,1.10,1.2,1.22,0.49,0.25,0.24,0.28,0.95]])

    def test_price_tracker_5050_no_rebalance(self):
        portfolios = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)
        sim_tracker = simulator.SimulationTracker(self.time_series,portfolios,self.split_params)
        sim_tracker.run_simulations()

        # Create an instance of the ReturnsCalculator class
        calculator = analysis.ReturnsCalculator(sim_tracker.allocated_capital)
        
        # Run the calculate_returns method
        calculator.calculate_returns()

        
        self.assertTrue(np.allclose(calculator.sim_portfolio, self.expected_portfolio_5050_no_rebalance))
        
    def test_price_tracker_5050_rebalance_down(self):
        portfolios = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)
        sim_tracker = (simulator
                        .SimulationTracker(self.time_series_sudden_drop,portfolios,self.split_params)
                        .run_simulations()
                        )

        calculator = (analysis.ReturnsCalculator(sim_tracker.allocated_capital)
                        .calculate_returns()
                        )
        self.assertTrue(np.allclose(calculator.sim_portfolio, self.expected_portfolio_5050_sudden_drop_rebalance_50pct))

    def test_price_tracker_5050_rebalance_up(self):
        portfolios = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.split_params)
        sim_tracker = (simulator
                        .SimulationTracker(self.time_series_sudden_up,portfolios,self.split_params)
                        .run_simulations()
                        )

        calculator = (analysis.ReturnsCalculator(sim_tracker.allocated_capital)
                        .calculate_returns()
                        )
        self.assertTrue(np.allclose(calculator.sim_portfolio, self.expected_portfolio_5050_sudden_up_rebalance_150pct))

    def test_call_option_write(self):
        traders = initialize_executors(n=1,return_function_params=self.ts_params,strategy_params=self.options_params)
        sim_tracker = (simulator
                        .SimulationTracker(self.time_series_sudden_drop_and_bounce_back,traders,self.options_params)
                        .run_simulations(logs_dir=env.TESTS_FOLDER)
                        )
        calculator = (analysis.ReturnsCalculator(sim_tracker.allocated_capital)
                        .calculate_returns()
                        )
        
        cum_returns = calculator.sim_cum_retuns
        pass
class TestDecigionLogic(unittest.TestCase):
    def test_threshold_below(self):
        
        self.assertFalse(simulator.check_is_below_threshold(current_price=1,prev_price=0.5,threshold=0.5))
        self.assertTrue(simulator.check_is_below_threshold(current_price=0.49,prev_price=1,threshold=0.5))

    def test_threshold_above(self):
        
        self.assertFalse(simulator.check_is_above_threshold(current_price=0.49,prev_price=1,threshold=0.5))
        self.assertTrue(simulator.check_is_above_threshold(current_price=1,prev_price=0.5,threshold=0.5))


class TestBSMOptionPricer(unittest.TestCase):
    def test_price_call(self):
        pass
    def test_price_put(self):
        pass

if __name__ == '__main__':
    unittest.main()