
import numpy as np

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
    def calc_avg_sharpe(self,ts):
        mean_v = ts.mean(axis=1)
        std_v = ts.std(axis=1)
        sharpe_v = ((mean_v - self.risk_free_rate/365) / std_v).mean()
        return sharpe_v
    def calculate_stats(self):
        self._stats["P(losing <50%)"] = (self.sim_cum_retuns[:, -1] >= 0.5).mean().mean()
        self._stats["P(losing <30%)"] = (self.sim_cum_retuns[:, -1] >= 0.7).mean().mean()
        self._stats["P(gaining 60%)"] = (self.sim_cum_retuns[:, -1] >= 1.6).mean().mean()
        
        E_R = (self.sim_cum_retuns[:, -1]-1).mean()
        std_ = (self.sim_cum_retuns[:, -1]-1).std() / np.sqrt(self.sim_cum_retuns.shape[0])

        self._stats["E(R)"] = E_R
        self._stats["Sharpe"] = round( (E_R - self.risk_free_rate) / std_, 3)
        # self._stats["Sharpe"] = self.calc_avg_sharpe(self.sim_retuns)
        self._stats[f"Daily {100-self.confidence_level}% VaR"] = np.percentile(self.sim_retuns, self.confidence_level, axis=1).mean()
        self._stats[f"Max Total VaR"] = (self.sim_cum_retuns[:, -1]-1).min()
        self._stats[f"Total {100-self.confidence_level}% VaR"] = np.percentile(self.sim_cum_retuns[:, -1]-1, self.confidence_level)
        return self

    @property
    def stats(self):        
        return self._stats
        
    @property
    def stats_str(self):
        return '\n'+'\n'.join([f'{k}: {round(v,3)}' if ('P(' not in k) and ('VaR' not in k) and ('E(R)' not in k) else f'{k}: {round(100*v,3)}%'
            
            for k,v in self._stats.items()])
