import numpy as np
import random
from nqdm import nqdm
def random_return(price, t, params):
    return random.gauss(params['mu'], params['sigma'])

def generate_time_series(N: int, T: int, current_price:float,return_func, params, ):
    """
    Generates N time series using the return function provided and saves them to file_path if provided.
    :param N: number of time series to generate
    :param T: number of time steps in each series
    :param return_func: function to generate returns for each time step. It should take in 2 parameters:
                    1) current price
                    2) time step
                    and return the return for the next step
    :param params: parameter for the return function
    :return: generated time series
    """
    time_series = np.zeros((N, T))
    time_series[:,0] = current_price
    print('running simulations..')
    for i in nqdm(range(N)):
        for j in range(1,T):
            time_series[i,j] = time_series[i,(j-1)] * (1. + return_func(current_price, j, params))
            if time_series[i,j] < 0.:
                time_series[i,j] = 0.
    return time_series