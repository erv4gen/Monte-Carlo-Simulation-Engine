import pandas as pd
from asyncio.proactor_events import constants
import numpy as np
from . import assets, constants
from typing import Dict
class ReturnsCalculator:
    def __init__(self, allocated_capital: np.ndarray, confidence_level: int = 5,risk_free_rate:float=0.01):
        '''
        `allocated_capital` is expected to be a numpy array with (n,t,k) shape, where
        n: number of sims
        t: number of timestamps
        k: number of assets in the portfolio
        '''
        self.allocated_capital = allocated_capital
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self._stats = {}
        self._calc_portfolio()

    def _calc_portfolio(self):
        self.sim_portfolio = np.nan_to_num(self.allocated_capital).sum(axis=2)
    def calculate_returns(self):
        self.sim_retuns = np.diff(self.sim_portfolio, axis=1) / self.sim_portfolio[:, :-1]
        self.sim_retuns = np.nan_to_num(np.insert(self.sim_retuns, 0, 0, axis=1))

        self.sim_cum_retuns = np.cumprod(self.sim_retuns + 1, axis=1)
        
        return self
    
    def _calc_sharpe(self,r,std):
        return ((r - self.risk_free_rate/constants.AnnualTimeInterval.days.value) / std).mean()
    def calc_avg_sharpe(self,ts):
        mean_v = ts.mean(axis=1)
        std_v = ts.std(axis=1)
        # sharpe_v = ((mean_v - self.risk_free_rate/constants.AnnualTimeInterval.days.value) / std_v).mean()
        sharpe_v = self._calc_sharpe(r=mean_v,std=std_v).mean()
        return sharpe_v
    
    def calculate_sample_stats(self):
        """
        Take a sample of the portfolio stochastic process and calculate performance of the time series.

        """
        portfolio = self.sim_portfolio[0,:]
        # Calculate returns
        returns = np.diff(portfolio) / portfolio[:-1]

        # Total return
        total_return = portfolio[-1] / portfolio[0] - 1

        # Annualized Return
        annualized_return = (1 + total_return)**(1/(len(portfolio)/constants.AnnualTimeInterval.days.value)) - 1  

        # Volatility
        volatility = np.std(returns)

        # Annualized Volatility
        annualized_volatility = volatility * np.sqrt(constants.AnnualTimeInterval.days.value) 

        # Sharpe Ratio
        sharpe_ratio = self._calc_sharpe(r=annualized_return,std=annualized_volatility).mean()

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns)
        annualized_downside_deviation = downside_deviation * np.sqrt(constants.AnnualTimeInterval.days.value) 
        sortino_ratio = (annualized_return - self.risk_free_rate) / annualized_downside_deviation

        self._sample_stats = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio
        }
        return self
    
    def calculate_stats(self):
        self._stats["P(losing <50%)"] = (self.sim_cum_retuns[:, -1] >= 0.5).mean().mean()
        self._stats["P(losing <30%)"] = (self.sim_cum_retuns[:, -1] >= 0.7).mean().mean()
        self._stats["P(gaining 60%)"] = (self.sim_cum_retuns[:, -1] >= 1.6).mean().mean()
        
        T = self.sim_cum_retuns.shape[1]
        E_R = (self.sim_cum_retuns[:, -1]-1).mean()
        E_R_anulz = ( (self.sim_cum_retuns[:, -1]-1 ) * (constants.AnnualTimeInterval.days.value/T) ).mean()
        std_ = (self.sim_cum_retuns[:, -1]-1).std() / np.sqrt(T)

        self._stats["E(R)"] = E_R
        self._stats["E(R_annualized)"] = E_R_anulz
        # self._stats["Sharpe"] = round( (E_R - self.risk_free_rate) / std_, 3)
        self._stats["Sharpe"] = self.calc_avg_sharpe(self.sim_retuns)
        self._stats[f"Daily {100-self.confidence_level}% VaR"] = np.percentile(self.sim_retuns, self.confidence_level, axis=1).mean()
        self._stats[f"Max Total VaR"] = (self.sim_cum_retuns[:, -1]-1).min()
        self._stats[f"Total {100-self.confidence_level}% VaR"] = np.percentile(self.sim_cum_retuns[:, -1]-1, self.confidence_level)
        return self

    @property
    def stats(self):        
        return self._stats
    
    @property
    def sample_stats(self):
        return self._sample_stats
    
    def _format_values(self)->Dict:
        return {k: str(round(v,3))  if ('P(' not in k) and ('VaR' not in k) and ('E(R' not in k) 
                else str(round(100*v,3))+'%'
            
            for k,v in self._stats.items()}
    @property
    def stats_str(self):
        return '\n'+'\n'.join([f'{k}:{v}' for k,v in self._format_values().items()] )

    @property
    def stats_df(self):
        return pd.DataFrame(self._format_values().items(),columns=['Metric','Value'])
