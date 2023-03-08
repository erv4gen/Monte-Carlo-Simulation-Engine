from typing import NamedTuple
import json
import pandas as pd
from typing import Tuple
import datetime as dt
import os
from .analysis import ReturnsCalculator
import logging
from dataclasses import dataclass

PRICE_MODEL_DICT = {'log_normal_return':'Lognormal Random Walk'}

class ComparisonAnnotation:
    def __init__(self,sigma,price_model:str,n_sims:int,n_steps:int,benchmark:str,percent_allocated:float=1.0,rebalance_events:str='',cash_interest:float=0.0,staking_rate:float=0.0,option_range:str='',stats:str=None) -> None:
        self.sigma =sigma
        self.price_model =price_model
        self.n_sims=n_sims
        self.n_steps= n_steps
        self.benchmark = benchmark
        self.percent_allocated = percent_allocated
        self.rebalance_events = rebalance_events
        self.cash_interest=cash_interest
        self.staking_rate=staking_rate
        self.option_range = option_range
        self.stats=stats

    def render_param_str(self)->str:
        price_model_frt = PRICE_MODEL_DICT[self.price_model]
        base_str = f'Strategy Params\nAsset price model: {price_model_frt}\n#Sims: {self.n_sims}\n#Time intervals: {self.n_steps},days\nAsset volatility: {round(100*self.sigma)}%'
        benchmark_str = f'\nBenchmark: {self.benchmark}' if self.benchmark!='' else ''
        percent_allocated_str = f'\nCash allocated: {round(100*self.percent_allocated)}%'
        rebalance_events_str = f'\nRebalance when: {self.rebalance_events}' if self.percent_allocated<1.0 else ''
        cash_str= f'\nCash interest: {round(100* self.cash_interest)}%' if self.cash_interest>0.0 else ''
        stake_str = f'\nStaking rate: {round(100*self.staking_rate)}%' if self.staking_rate>0.0  else ''
        option_str = f'\n{self.option_range}'

        
        return base_str + benchmark_str+percent_allocated_str+rebalance_events_str+cash_str + stake_str + option_str

    def render_stats_str(self)->str:
        stats_str = f'Strategy Result Stats'+self.stats if self.stats is not None else ''
        return stats_str

def create_logger(log_file:str=None):
    if log_file is not None:
        logger = logging.getLogger(__name__)
        logger.disabled = False
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger = logging.getLogger(__name__)
        logger.disabled = True
    
    return logger

class StrategyParams(NamedTuple):
    amount_multiple: float = 1.0
    percent_allocated:float= 1.0
    rebalance_threshold_down: float= 0.00
    rebalance_threshold_up: float= 1e10
    max_rebalances:int= 0
    cash_interest:float= 0.0
    coin_interest:float= 0.0
    rebalance_every: int = 1e10
    option_every_itervals:int = 1e10
    option_duration:int = 1e10
    option_amount_pct_of_notional:float = 0.50
    option_straddle_pct_from_strike: float = 0.1
    ticker_name : str = 'ETH'
    benchmark_strategy_name: str = 'Buy and Hold'

class Config(NamedTuple):
    return_function_params: dict
    strategy_function_params: dict
    return_function: str
    plot_params: dict
    save_logs:bool=False
    logs_dir:str = None

    def __str__(self):
        return json.dumps(self._asdict(), indent=4)

    def to_dict(self):
        return {k:v for k,v in self._asdict().items() if v is not None}
def read_config(config_file: str) -> Config:
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    return Config(**config_data)


def save_config_to_csv(config: Tuple, path: str):
    config_dict = config._asdict()
    df = pd.DataFrame(config_dict, index=[0])
    df.to_csv(path, index=False)


class Env:
    def __init__(self):
        self.timestp_ = dt.datetime.now().strftime('%Y%m%d%H%M%S')
        self.SIM_FOLDER = os.path.join(os.path.abspath('.'),'data','runs', self.timestp_)
        self.LOGS_FOLDER = os.path.join(self.SIM_FOLDER,'logs')
        self.TESTS_FOLDER = os.path.join(os.path.abspath('.'),'data','tests', self.timestp_)
        self.paths = {
            'TS_SIMS': 'prices_sims.pkl',
            'TS_PORTFLO_SIM': 'portfolio_sims.pkl',
            'PLOT_TS': 'prices_sims.png',
            'PLOT_PORTFOLIO': 'portfolio_sims.png',
            'PLOT_COMPARISON':'comparison.png',
            'PLOT_BASELINEONLY':'baseline_only.png',
            'PLOT_HISTOGRAMS':'histograms.png',
            'STATS_CSV': 'portfolio_summary.csv',
            'CONFIG_CSV': 'simulation_params.csv',

        }
        
        print('simulation will be saved to ', self.SIM_FOLDER)

    def create_run_env(self):
        if not os.path.exists(self.LOGS_FOLDER):
            os.makedirs(self.LOGS_FOLDER)
        return self
    def create_test_env(self):
        if not os.path.exists(self.TESTS_FOLDER):
            os.makedirs(self.TESTS_FOLDER)
        return self
    def __getattr__(self, name):
        if name in self.paths:
            return os.path.join(self.SIM_FOLDER, self.paths[name])
        else:
            raise AttributeError("Env has no attribute {}".format(name))




def save_stats_to_csv(return_calculator:ReturnsCalculator, path:str):
    df = pd.DataFrame.from_dict(return_calculator.stats,orient='index',columns=['value'])
    df.to_csv(path)


def config_sanity_check(config):
    assert config.return_function_params['N']> 1, 'N must be > 1'