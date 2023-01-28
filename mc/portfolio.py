import numpy as np
import pandas as pd
from typing import Tuple , Dict

from mc import series_gen

def run_one_asset_rebalance_portfolio(time_series: np.ndarray, 
                        percent_allocated: float, 
                        threshold: float,
                        k: int) -> np.ndarray:
    """
    Rebalances the portfolio so that the proportion of capital allocated to the asset
    is always `percent_allocated` after the price of the asset drops by more than `threshold`
    since the last rebalancing. 
    Rebalancing is done at most k times.
    """
    n, t = time_series.shape
    rebalancing_count = np.zeros((n,)) # to keep track of how many times rebalancing has been done
    last_rebalanced_price = np.copy(time_series[:,0]) # to keep track of the last rebalanced price

    
    capital_in_asset = time_series[:,0] * percent_allocated
    capital_in_cash = time_series[:,0] * (1-percent_allocated)
    allocated_capital = np.stack(( capital_in_asset[:,np.newaxis]
                                ,capital_in_cash[:,np.newaxis]
                                ),axis=2
                                )
    allocated_capital = np.repeat(allocated_capital,t,axis=1)

    for i in range(n):
        for j in range(1, t):
            payoff = (time_series[i, j]/time_series[i, j-1]  )
            allocated_capital[i,j,0] = allocated_capital[i,j-1,0] * payoff
            if time_series[i, j] / last_rebalanced_price[i] < 1 - threshold and rebalancing_count[i] < k:
                
                allocated_capital[i,j:,:] = [allocated_capital[i,j].mean()
                                        ,allocated_capital[i,j].mean()]

                rebalancing_count[i] += 1
                last_rebalanced_price[i] = time_series[i, j]

    return allocated_capital


class ReturnsCalculator:
    def __init__(self, allocated_capital: np.ndarray, confidence_level: int = 5):
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
        self._stats["P-not losing 50%"] = (self.sim_cum_retuns[:, -1] >= 0.5).mean()
        self._stats["P-gaining 60%"] = (self.sim_cum_retuns[:, -1] >= 1.6).mean()
        self._stats["VAR"] = np.percentile(self.sim_retuns, self.confidence_level, axis=1)
        return self
        
    @property
    def stats(self):
        return self._stats


def save_stats_to_csv(return_calculator:ReturnsCalculator, path:str):
    df = pd.DataFrame(return_calculator.stats)
    df.to_csv(path)