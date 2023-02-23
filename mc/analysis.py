
import numpy as np

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
        self.sim_portfolio = np.nan_to_num(self.allocated_capital).sum(axis=2)
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
