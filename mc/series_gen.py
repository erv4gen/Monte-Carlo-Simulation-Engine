import numpy as np
import random
from nqdm import nqdm
from typing import List
import math
from datetime import datetime, timedelta


# def random_return(price, t, params):
#     return price * (1+ random.gauss(params['mu'], params['sigma']))


def random_return(price, t,T, params):
    r = params.get("r", 0)
    sigma = params.get("sigma", 1) 
    return price * (1+ r/T + sigma/(T**0.5) * random.gauss(0, 1))

def log_normal_return(price, t, T,params):
    mu = params.get("mu", 0)
    sigma = params.get("sigma", 1)
    # dt = params.get("dt", 1)
    return price * (np.exp(mu + np.random.normal(0, sigma / np.sqrt(T))) )




RETURN_FUNCTIONS = {'Lognormal Random Walk':log_normal_return
                        ,'Normal Random Walk':random_return}


def return_functions(function_name):
    return RETURN_FUNCTIONS[function_name]




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
    print('simulating prices..')
    for i in nqdm(range(N)):
        for j in range(1,T):
            time_series[i,j] = return_func(time_series[i,(j-1)], j,T, params)
            if time_series[i,j] < 0.:
                time_series[i,j] = 0.
    return time_series





def cash_investment(n: int, initial_amount: float, rate: float, capitalization_period: int) -> List[float]:
    """
    Generate a time series of an investment with an initial amount, a fixed rate, and a fixed capitalization period.

    Args:
        n (int): The length of the time series in days.
        initial_amount (float): The initial investment amount.
        rate (float): The annual interest rate as a decimal.
        capitalization_period (int): The number of days between interest payments.

    Returns:
        List[float]: A list of n floats representing the value of the investment on each day of the time series.
    """
    daily_rate = math.pow(1 + rate, 1/365) - 1
    capitalization_multiplier = math.pow(1 + daily_rate, capitalization_period)
    daily_multiplier = math.pow(1 + daily_rate, 1)
    values = [initial_amount]
    
    for i in range(1, n):
        last_capitalization_index = ((i - 1) // capitalization_period) * capitalization_period
        days_since_last_capitalization = i - last_capitalization_index

        if days_since_last_capitalization >= capitalization_period:
            values.append(values[-1] * capitalization_multiplier)
        else:
            values.append(values[-1] * daily_multiplier)

    return values