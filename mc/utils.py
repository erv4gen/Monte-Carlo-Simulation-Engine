from typing import NamedTuple
import json
import pandas as pd
from typing import Tuple
import datetime as dt
import os

class StrategyParams(NamedTuple):
    amount_multiple: float = 1.0
    percent_allocated:float= 1.0
    rebalance_asset_ration: float = 0.5
    rebalance_threshold_down: float= 0.00
    rebalance_threshold_up: float= 1e10
    max_rebalances:int= 0
    cash_interest:float= 0.0
    rebalance_every: int = 366
    option_premium:float = 0.03
    option_every_itervals:int = 365 
    option_duration:int = 365 
    option_amount_pct_of_notional:float = 0.25

class Config(NamedTuple):
    return_function_params: dict
    strategy_function_params: dict
    return_function: str

    def __str__(self):
        return json.dumps(self._asdict(), indent=4)

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
        self.SIM_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs', self.timestp_)
        self.paths = {
            'TS_SIMS': 'prices_sims.pkl',
            'TS_PORTFLO_SIM': 'portfolio_sims.pkl',
            'PLOT_TS': 'prices_sims.png',
            'PLOT_PORTFOLIO': 'portfolio_sims.png',
            'STATS_CSV': 'portfolio_summary.csv',
            'CONFIG_CSV': 'simulation_params.csv',
        }
        if not os.path.exists(self.SIM_FOLDER):
            os.makedirs(self.SIM_FOLDER)

        print('simulation will be saved to ', self.SIM_FOLDER)

    def __getattr__(self, name):
        if name in self.paths:
            return os.path.join(self.SIM_FOLDER, self.paths[name])
        else:
            raise AttributeError("Env has no attribute {}".format(name))