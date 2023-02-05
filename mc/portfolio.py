from ast import Eq
from locale import currency
import numpy as np
import pandas as pd
from typing import Tuple , Dict
from tqdm import tqdm
from mc import series_gen
from .utils import StrategyParams

class NegativePriceExepton(Exception):
    def __init__(self, *args: object, **kwargs:object) -> None:
        super().__init__(*args,**kwargs)

    def __str__(self) -> str:
        return 'Negative price setter is not allowed'

class Asset:
    def __init__(self,amount:float=0.0,initial_price:float=1.0) -> None:
        self._amount = amount
        #initial price
        self._s0_price = initial_price
        #current price (last recorded)
        self._st_price = initial_price
    

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self,value:float):
        assert value >0., NegativePriceExepton()
        self._amount = value

    @property
    def initial_value(self):
        value_ = self._amount * self._s0_price
        return value_

    @property
    def value(self):
        value_ = self._amount * self._st_price
        return value_

    @property
    def current_price(self):
        return self._st_price
    
    @current_price.setter
    def current_price(self,current_price):
        self._st_price = current_price        

    def pct_return(self):
        return 0.0 if self._s0_price==0.0 else self._s1_price/ self._s0_price -1.

class Cash(Asset):
    def __init__(self, *args: object, **kwargs:object) -> None:
        super().__init__(*args,**kwargs)

class Equity(Asset):
    def __init__(self, *args: object, **kwargs:object) -> None:
        super().__init__(*args,**kwargs)

class Option(Asset):
    def __init__(self,premium_pct:float,*args, **kwargs) -> None:
        '''
        `premium_pct` what is the premium as a pct of the price
        '''
        super().__init__(*args,**kwargs)
        self._premium_pct = premium_pct
        #mark if the option is active
        self._ALIVE = False
        self._strike = None
    def write(self,current_price:float,strike:float,amount:float):
        '''
        When option is written, it becomes alive until assigmen
        the function returns `premium_value`
        '''
        if self._ALIVE: return None
        premium_value = current_price * self._premium_pct
        self._strike = strike
        self.amount = amount
        self._ALIVE = True
        return premium_value

    def assign(self):
        '''
        Assigment makes option not alive and return the delivery asset
        '''
        if not self._ALIVE: return None

        assets_to_delivery = self.amount * self._strike
        self._ALIVE = False
        return assets_to_delivery


class CallOption(Option):
    def __init__(self, premium_pct: float) -> None:
        super().__init__(premium_pct)
    
    def assign(self,current_price:float):
        assets_to_delivery = super().assign()
        #if option in the monay
        if current_price>= self._strike:
            return assets_to_delivery

class PutOption(Option):
    def __init__(self, premium_pct: float) -> None:
        super().__init__(premium_pct)
    
    def assign(self,current_price:float):
        assets_to_delivery = super().assign()
        #if option in the monay
        if current_price <= self._strike:
            return assets_to_delivery


class Portfolio:
    def __init__(self) -> None:
        self._cash: Cash = None
        self._equity: Equity = None
        self._option: Option = None
    
    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self,value:Cash):
        self._cash = value

    @property
    def equity(self):
        return self._equity

    @equity.setter
    def equity(self,value:Equity):
        self._equity = value


    @property
    def option(self):
        return self._option

    @option.setter
    def option(self,value:Option):
        self._option = value

    @property
    def capital(self):
        equity_value = self.equity.value
        cash_value = self.cash.value
        return equity_value + cash_value

def asset_return(asset:Asset,price:float):
    '''
    Return the value of the `Asset` in the numeraire 
    '''
    asset.current_value = price
    return asset.pct_return()


   #TODO
    #Split the array into multiple object each represent an asset with each own associated time series.
    # access each asset independently
    # probably have a function that will balance assets between 
    # spin off the rebalancing logic into a separate function
    # 

def initialize_portfolios(n,initial_price,strategy_params: StrategyParams):
    '''
    Return a list of portfolios for each simulation
    '''
    sim_portfolios = []
    for i in range(n):
        portfolio = Portfolio()
        portfolio.cash = Cash(amount =  strategy_params.amount_multiple * (1-strategy_params.percent_allocated) )
        portfolio.equity = Equity(amount= strategy_params.amount_multiple * strategy_params.percent_allocated
                                ,initial_price=initial_price)
        sim_portfolios.append(portfolio)

    return sim_portfolios

def run_one_asset_rebalance_option(time_series: np.ndarray, 
                        strategy_params: StrategyParams
                        ) -> np.ndarray:

    n, t = time_series.shape
    
    

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