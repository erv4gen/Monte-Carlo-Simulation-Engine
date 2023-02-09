from ast import Eq
from locale import currency
import re
import numpy as np
import pandas as pd
from collections.abc import Mapping

from tqdm import tqdm
from .collections import *
from .assets import *
from .utils import StrategyParams , Config 
from typing import List, Dict



def check_is_below_threshold(current_price:float,prev_price:float,threshold:float):
    return  (current_price / prev_price) < (1 - threshold)

def check_is_above_threshold(current_price:float,prev_price:float,threshold:float):
    return  (current_price / prev_price) > (1 + threshold)

class Trader:
    def __init__(self, portfolio: Portfolio):
        self._portfolio = portfolio

    def buy_equity(self, asset: Asset,amount:float,transaction_price:float=None) -> None:
        '''
        This function is an adjustment transaction for an asset up
        '''
        transaction_price = self.equity.current_price if transaction_price is None else transaction_price

        cost = amount * transaction_price

        if cost > amount: raise NotEnoughMoney()

        new_cost_average_price = weighted_avg(x1= self.equity.initial_price,x2=transaction_price,w1=self.equity.amount,w2=amount
        )
        #withdraw cash
        self._portfolio.cash.amount -= cost

        #add equity
        self._portfolio.equity.initial_price = new_cost_average_price
        self._portfolio.equity.amount += asset.amount

    def sell_equity(self, asset: Asset,amout:float,transaction_price:float = None) -> None:
        transaction_price = asset.current_price if transaction_price is None else transaction_price

        if amout > self._portfolio._equity.amount: raise NotEnoughAmount()
        
        asset.amount -= asset.amount
        self._portfolio._cash.amount += asset.amount * transaction_price



    def short_sell(self, asset: Asset,amout:float,transaction_price:float=None) -> None:
        transaction_price = asset.current_price if transaction_price is None else transaction_price

        if self._portfolio.cash < amout * transaction_price: raise NotEnoughMoney()
        asset.amount -= asset.amount
        self._portfolio._cash.amount += asset.amount * transaction_price
        
    def execute_trade(self, asset: Asset, amount: float, transaction_type: TransactionType,transaction_price:float=None) -> None:
        if transaction_type == TransactionType.BUY:
            self.buy_asset(asset, amount,transaction_price)
        elif transaction_type == TransactionType.SELL:
            self.sell_asset(asset, amount,transaction_price)

        elif transaction_type == TransactionType.SHORT_SELL:
            self.short_sell(asset, amount,transaction_price)
        else:
            raise ValueError("Invalid transaction type: {}".format(transaction_type))


    def rebalance(self,asset: Asset,target_share:float) -> None:
        '''
        rebalance cash and asset.
        `target_share`: result asset share in the portfolio
        '''
        target_asset_value = self.capital * target_share

        target_asset_amount =  target_asset_value / self.equity.current_price
        amout_diff = target_asset_amount - self.equity.amount
        if amout_diff > 0:
            # buy more equity 
            self.execute_trade(asset, amout_diff,TransactionType.BUY)
            
        else:
            # sell some equity
            self.execute_trade(asset, amout_diff * -1.0,TransactionType.SELL)
            

    def write_options(self,t,price,amount) -> None:

        #update option status and add cash
        call_strike = price * 1.1
        self.call_option.write(price,call_strike,amount,t)

        put_strike = price * 0.9
        self.put_option.write(price,put_strike,amount,t)

    def option_assigment(self,t,price) -> None:
        '''
        Check of any Options are due on assigment and execute either sell or buy operation
        '''
        if self._call_option is not None and self._call_option.decay(t):
            asset_delivery =self._call_option.assign(price)
            self.sell_equity(amount= asset_delivery.amount ,transaction_price=asset_delivery.current_price)
        
        elif self._put_option is not None and self._put_option.decay(t):
            asset_delivery =self._put_option.assign(price)
            self.buy_equity(adj_amount=asset_delivery.amount ,transaction_price=asset_delivery.current_price)


class SimulationTracker:
    def __init__(self,time_series:np.array,portfolios: List[Portfolio] ,strategy_params:StrategyParams) -> None:
        self._ts = time_series
        self._portfolios = portfolios
        n,t =time_series.shape
        self._n = n
        self._t = t

        self._rebalancing_count = np.zeros((n,)) # to keep track of how many times rebalancing has been done
        self._last_rebalanced_price = np.copy(time_series[:,0]) # to keep track of the last rebalanced price

        self._ASSET_INDEX = {'equity':0,'cash':1}
        capital_in_asset = time_series[:,0] * strategy_params.percent_allocated
        capital_in_cash = time_series[:,0] * (1-strategy_params.percent_allocated)

        self.strategy_params = strategy_params
        self._allocated_capital = np.stack(( capital_in_asset[:,np.newaxis]
                                    ,capital_in_cash[:,np.newaxis]
                                    ),axis=2
                                    )

        self._allocated_capital = np.repeat(self._allocated_capital,t,axis=1)


        self._asset_threshold_down = None
        self._asset_threshold_up = None

    def _is_new_month(self,i):
        return i % 31 == 0

    def add_rebalance_below(self,threshold):
        self._asset_threshold_down = threshold
        return self
    
    def add_rebalance_above(self,threshold):
        self._asset_threshold_up = threshold
        return self

    def max_rebalances_cheker(self,i):
        return self._rebalancing_count[i] < self.strategy_params.max_rebalances

    def _get_price(self,i:int,j:int):
        return self._ts[i,j]
    
    def _capitalize_cash(self,i:int,j:int):
        asset_idx = self._ASSET_INDEX['cash']
        self._allocated_capital[i,j,asset_idx] =  self._allocated_capital[i,j-1,asset_idx] * (1+self.strategy_params.cash_interest/365)
    
    def _change_asset_price(self,i:int,price:float):
        
        self._portfolios[i].log_asset_price(price)

    def _log_equity_value(self,i:int,j:int):
        asset_idx = self._ASSET_INDEX['equity']
        self._allocated_capital[i,j,asset_idx]  =self._portfolios[i].equity.value
    
    def _log_cash_value(self,i:int,j:int):
        asset_idx = self._ASSET_INDEX['cash']
        self._allocated_capital[i,j,asset_idx]  = self._portfolios[i].cash.value


    def log_state_change(self,i:int,j:int):
        self._log_cash_value(i,j)
        self._log_equity_value(i,j)
    
    def _rebalance_portfolio(self,i,j,price):
        is_below_threshold_triggered = check_is_below_threshold(price,self._last_rebalanced_price[i],self._asset_threshold_down) if self._asset_threshold_down is not None else False
        is_above_threshold_triggered = check_is_above_threshold(price,self._last_rebalanced_price[i],self._asset_threshold_up) if self._asset_threshold_up is not None else False

        capped_threshold_rebalances = (is_below_threshold_triggered ^ is_above_threshold_triggered)  and self.max_rebalances_cheker(i)
        interval_rebalance = False
        if capped_threshold_rebalances or interval_rebalance:

            self._portfolios[i].rebalance(self.strategy_params.rebalance_asset_ration)
            self._rebalancing_count[i] += 1
            self._last_rebalanced_price[i] = price

            self.log_state_change(i,j)

    def _validate_derivatives(self,i,t,price):
        
        self._portfolios[i].option_assigment(t,price)

        amount = self._portfolios[i].equity._amount * self.strategy_params.option_amount_pct_of_notional
        self._portfolios[i].write_options(t= self.strategy_params.option_duration + t
                                        ,price=price,amount=amount)

        self.log_state_change(i,t)

    def run_simulations(self):
        '''
        Runner finction that exectute strategy for each price prajectory
        '''

        for i in tqdm(range(self._n)):
            for j in range(1, self._t):
                
                #get information from market
                new_price = self._get_price(i,j)
                prev_price =self._get_price(i,j-1)
                # payoff = (new_price/prev_price)
                
                #update portfolio state pre action
                self._change_asset_price(i,new_price)
                self._capitalize_cash(i,j)

                #assign market return to allocated portfolio
                self._log_equity_value(i,j)

                #check derivative contract execution
                self._validate_derivatives(i,j,new_price)

                #rebalance if needec
                self._rebalance_portfolio(i,j,new_price)
        return self

    @property
    def allocated_capital(self):
        return self._allocated_capital


def asset_return(asset:Asset,price:float):
    '''
    Return the value of the `Asset` in the numeraire 
    '''
    asset.current_value = price
    return asset.pct_return()


def initialize_executors(n,initial_price,strategy_params: StrategyParams) -> List[Trader]:
    '''
    Return a list of portfolios for each simulation
    '''

    sim_portfolios = []
    for i in range(n):
        portfolio = Portfolio()
        portfolio.cash = Cash(amount =  strategy_params.amount_multiple * (1-strategy_params.percent_allocated) * initial_price )
        asset = Equity(ticker=strategy_params.ticker_name,amount= strategy_params.amount_multiple * strategy_params.percent_allocated
                                ,initial_price=initial_price)

        portfolio.add_asset(asset)
        # portfolio.call_option = EuropeanNaiveCallOption(strategy_params.option_premium)
        # portfolio.put_option = EuropeanNaivePutOption(strategy_params.option_premium)

        trader = Trader(portfolio)

        sim_portfolios.append(trader)

    return sim_portfolios

def run_one_asset_rebalance_portfolio_v1(time_series: np.ndarray, 
                        strategy_params: StrategyParams,
                        config: Config

                        ) -> np.ndarray:

    n, t = time_series.shape
    #initiate portfolios
    sim_portfolios = initialize_executors(n
                                            ,initial_price = config.return_function_params['current_price']
                                            ,strategy_params=strategy_params
                                            )
    
    #walk throught time series 
    sim_tracker = (SimulationTracker(time_series,sim_portfolios,strategy_params)
                        .add_rebalance_below(0.5)
                        .run_simulations()
                        )
    return sim_tracker.allocated_capital

def run_one_asset_rebalance_portfolio_v0(time_series: np.ndarray, 
                        strategy_params: StrategyParams
                        ) -> np.ndarray:
    """
    Rebalances the portfolio so that the proportion of capital allocated to the asset
    is always `percent_allocated` after the price of the asset drops by more than `threshold`
    since the last rebalancing. 
    Rebalancing is done at most k times.
    """
    n, t = time_series.shape


    rebalancing_count = np.zeros((n,)) # to keep track of how many times rebalancing has been done
    last_rebalanced_price = np.copy(time_series[:,0]) # to keep track of the last rebalanced price

    
    capital_in_asset = time_series[:,0] * strategy_params.percent_allocated
    capital_in_cash = time_series[:,0] * (1-strategy_params.percent_allocated)

    allocated_capital = np.stack(( capital_in_asset[:,np.newaxis]
                                ,capital_in_cash[:,np.newaxis]
                                ),axis=2
                                )
    allocated_capital = np.repeat(allocated_capital,t,axis=1)

    print('running portfolio...')
    for i in tqdm(range(n)):
        for j in range(1, t):

            #assign market return to allocated portfolio
            payoff = (time_series[i, j]/time_series[i, j-1]  )
            allocated_capital[i,j,0] = allocated_capital[i,j-1,0] * payoff
            
            #add capitalization on the cash returns 
            allocated_capital[i,j,1] =  allocated_capital[i,j-1,1] * (1+strategy_params.cash_interest/365)

            if ((time_series[i, j] / last_rebalanced_price[i] < 1 - strategy_params.rebalance_threshold) \
                and (rebalancing_count[i] < strategy_params.max_rebalances)) \
                or (j % strategy_params.rebalance_every ==0):
                
                allocated_capital[i,j:,:] = [allocated_capital[i,j].mean()
                                        ,allocated_capital[i,j].mean()]

                rebalancing_count[i] += 1
                last_rebalanced_price[i] = time_series[i, j]
                continue

    return allocated_capital

