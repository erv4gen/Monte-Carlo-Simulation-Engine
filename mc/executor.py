from ast import Eq
from locale import currency
import numpy as np
import pandas as pd
from typing import Tuple , Dict
from tqdm import tqdm
from .assets import *
from .utils import StrategyParams , Config 
from typing import List


def check_is_below_threshold(current_price:float,prev_price:float,threshold:float):
    return  (current_price / prev_price) < (1 - threshold)

def check_is_above_threshold(current_price:float,prev_price:float,threshold:float):
    return  (current_price / prev_price) > (1 + threshold)

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

    def _rebalance_portfolio(self,i,j,price):
        is_below_threshold_triggered = check_is_below_threshold(price,self._last_rebalanced_price[i],self._asset_threshold_down) if self._asset_threshold_down is not None else False
        is_above_threshold_triggered = check_is_above_threshold(price,self._last_rebalanced_price[i],self._asset_threshold_up) if self._asset_threshold_up is not None else False

        capped_threshold_rebalances = (is_below_threshold_triggered ^ is_above_threshold_triggered)  and self.max_rebalances_cheker(i)
        interval_rebalance = False
        if capped_threshold_rebalances or interval_rebalance:
            self._portfolios[i].rebalance(self.strategy_params.rebalance_asset_ration)
            self._rebalancing_count[i] += 1
            self._last_rebalanced_price[i] = price

            self._log_cash_value(i,j)
            self._log_equity_value(i,j)

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


def initialize_portfolios(n,initial_price,strategy_params: StrategyParams) -> List[Portfolio]:
    '''
    Return a list of portfolios for each simulation
    '''
    sim_portfolios = []
    for i in range(n):
        portfolio = Portfolio()
        portfolio.cash = Cash(amount =  strategy_params.amount_multiple * (1-strategy_params.percent_allocated) * initial_price )
        portfolio.equity = Equity(amount= strategy_params.amount_multiple * strategy_params.percent_allocated
                                ,initial_price=initial_price)
        sim_portfolios.append(portfolio)

    return sim_portfolios

def run_one_asset_rebalance_option(time_series: np.ndarray, 
                        strategy_params: StrategyParams,
                        config: Config

                        ) -> np.ndarray:

    n, t = time_series.shape
    #initiate portfolios
    sim_portfolios = initialize_portfolios(n
                                            ,initial_price = config.return_function_params['current_price']
                                            ,strategy_params=strategy_params
                                            )
    
    #walk throught time series 
    for i in tqdm(range(n)):
        for j in range(1, t):

            pass

def run_one_asset_rebalance_portfolio(time_series: np.ndarray, 
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


class ReturnsCalculator:
    def __init__(self, allocated_capital: np.ndarray, confidence_level: int = 5):
        '''
        `allocated_capital` is expected to be a numpy array with (n,t,k) shape, where
        n: number of sims
        t: number of timestamps
        k: number of assets in the portfolio
        '''
        self.allocated_capital = allocated_capital
        self.confidence_level = confidence_level
        self._stats = {}
        self._calc_portfolio()

    def _calc_portfolio(self):
        self.sim_portfolio = self.allocated_capital.sum(axis=2)
    def calculate_returns(self):
        self.sim_retuns = np.diff(self.sim_portfolio, axis=1) / self.sim_portfolio[:, :-1]
        self.sim_retuns = np.insert(self.sim_retuns, 0, 0, axis=1)

        self.sim_cum_retuns = np.cumprod(self.sim_retuns + 1, axis=1)
        
        return self
    def calculate_stats(self):
        self._stats["P-not losing 50%"] = (self.sim_cum_retuns[:, -1] >= 0.5).mean().mean()
        self._stats["P-gaining 60%"] = (self.sim_cum_retuns[:, -1] >= 1.6).mean().mean()
        self._stats["VAR"] = np.percentile(self.sim_retuns, self.confidence_level, axis=1).mean()
        return self
        
    @property
    def stats(self):
        return self._stats


def save_stats_to_csv(return_calculator:ReturnsCalculator, path:str):
    df = pd.DataFrame.from_dict(return_calculator.stats,orient='index',columns=['value'])
    df.to_csv(path)