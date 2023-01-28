from typing import NamedTuple
import json
import pandas as pd
from typing import Tuple
import datetime as dt
import os


class Config(NamedTuple):
    N: int
    T: int
    return_function_params: dict
    percent_allocated: float
    rebalance_threshold: float
    max_rebalances: int
    current_price: int



def parse_config(config_file: str) -> Config:
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    return Config(
        N=config_data['N'],
        T=config_data['T'],
        return_function_params=config_data['return_function_params'],
        percent_allocated=config_data['percent_allocated'],
        rebalance_threshold=config_data['rebalance_threshold'],
        max_rebalances=config_data['max_rebalances'],
        current_price =config_data['current_price'],
    )


def save_config_to_csv(config: Tuple, path: str):
    config_dict = config._asdict()
    df = pd.DataFrame(config_dict, index=[0])
    df.to_csv(path, index=False)


class Env:
    def __init__(self):
        self.timestp_ = dt.datetime.now().strftime('%Y%m%d%H%M%S')
        self.SIM_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs', self.timestp_)
        self.paths = {
            'TS_SIMS': 'ts_sims.pkl',
            'TS_PORTFLO_SIM': 'ts_sims.pkl',
            'PLOT_TS': 'ts_sims.png',
            'PLOT_PORTFOLIO': 'ts_sims.png',
            'STATS_CSV': 'portfolio_summary.csv'
        }
        if not os.path.exists(self.SIM_FOLDER):
            os.makedirs(self.SIM_FOLDER)

        print('simulation will be saved to ', self.SIM_FOLDER)

    def __getattr__(self, name):
        if name in self.paths:
            return os.path.join(self.SIM_FOLDER, self.paths[name])
        else:
            raise AttributeError("Env has no attribute {}".format(name))