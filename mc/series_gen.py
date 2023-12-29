import numpy as np
import random
from nqdm import nqdm
from typing import List
import math
from datetime import datetime, timedelta
from scipy.stats import norm, invgamma

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


def generalized_hyperbolic_return(price, t, T, params):
    # Extract GH distribution parameters from params
    mu = params.get("mu", 0)
    alpha = params.get("alpha", 1)
    beta = params.get("beta", 0)
    delta = params.get("delta", 1)
    lambda_ = params.get("lambda_", -0.5)

    # Step 1: Generate a random variable from the inverse gamma distribution
    gamma_var = invgamma.rvs(a=-lambda_, scale=delta**2/alpha, size=1)[0]

    # Step 2: Generate a random variable from the normal distribution
    z = norm.rvs(loc=0, scale=np.sqrt(gamma_var), size=1)[0]

    # Step 3: Adjust the random variable for the GH distribution
    gh_var = mu + beta * gamma_var + z

    # Apply the GH distributed random variable to the price
    return price * (1 + gh_var)

RETURN_FUNCTIONS = {'Lognormal Random Walk':log_normal_return
                        ,'Normal Random Walk':random_return
                        ,"Generalized Hyperbolic":  generalized_hyperbolic_return
                        }



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





from typing import List
import math

def cash_investment(n: int, rate: float, initial_amount: float, capitalization_period: int):

    period_rate = rate * capitalization_period/365.25  # convert annual rate to period
    capitalization_multiplier = 1 + period_rate    
    capital_series = np.zeros(n)
    capital_series[0] = initial_amount
    
    for i in range(1,n):
        if i % capitalization_period ==0:
            capital_series[i] = capital_series[i-1] * capitalization_multiplier
        else:
            capital_series[i] = capital_series[i-1]
        
    return capital_series