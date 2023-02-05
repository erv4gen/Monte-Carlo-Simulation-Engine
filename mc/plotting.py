import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import seaborn as sns
from scipy import stats

def get_confidence_interval(ts,p):
    mean = ts.mean(axis=0)
    std = ts.std(axis=0)
    z = norm.ppf(p)
    lower_bound = mean - z * std 
    upper_bound = mean + z * std
    return lower_bound, upper_bound

def save_plot(fig,file_name):
    fig.savefig(file_name)


def plot_simulations(ts,params):
    fig, ax = plt.subplots()
    for i in range(ts.shape[0]):
        plt.plot(ts[i,:],alpha = params['plot']['alpha'],zorder =1
        ,linewidth=0.3
        )
    lower_bound, upper_bound = get_confidence_interval(ts,p=params['ci'])

    ax.fill_between(np.arange(ts.shape[1]), lower_bound, upper_bound, color='gray', alpha=0.7,zorder=2)

    ax.axhline(0, color='black', lw=1,linestyle=':')
    


    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])
    
    plt.show()
    
    return fig


def plot_histogram(ts,params):
    '''
    Plot histogram of the time series
    '''
    log_multi_period_prices = ts[:,-1]
    log_multi_period_rets = [x/params['starting_price'] - 1 for x in log_multi_period_prices]
    log_multi_period_rets_log = [np.log(x/params['starting_price']) for x in log_multi_period_prices]

    # plot the distribution of final prices
    # create the figure
    fig, ax = plt.subplots(nrows=3, figsize=(12,36)
    )

    # fit a dist plot
    sns.distplot(log_multi_period_prices, fit=stats.lognorm, ax=ax[0])
    sns.distplot(log_multi_period_rets, fit=stats.lognorm, ax=ax[1])
    sns.distplot(log_multi_period_rets_log, fit=stats.lognorm, ax=ax[2])

    # compute the params of the fitted normal curve
    (mu, s) = stats.norm.fit(log_multi_period_prices)
    ax[0].set_title('Distribution of Multi Period Geo Prices')
    ax[0].set_xlabel('Prices')
    ax[0].set_ylabel('Density')
    ax[0].set_xticklabels(['{:,.1f}'.format(x) for x in ax[0].get_xticks()])
    ax[0].legend(["normal dist. fit ($\mu=${0:,.2f}, $\sigma=${1:,.2f})".format(mu, s)])

    # compute the params of the fitted normal curve
    (mu, s) = stats.norm.fit(log_multi_period_rets)
    ax[1].set_title('Distribution of Multi Period Simple Returns')
    ax[1].set_xlabel('Returns')
    ax[1].set_ylabel('Density')
    ax[1].set_xticklabels(['{:,.1%}'.format(x) for x in ax[1].get_xticks()])
    ax[1].legend(["normal dist. fit ($\mu=${0:,.2%}, $\sigma=${1:,.2%})".format(mu, s)])

    # compute the params of the fitted normal curve
    (mu, s) = stats.norm.fit(log_multi_period_rets_log)
    ax[2].set_title('Distribution of Multi Period Log Returns')
    ax[2].set_xlabel('Returns')
    ax[2].set_ylabel('Density')
    ax[2].set_xticklabels(['{:,.1%}'.format(x) for x in ax[2].get_xticks()])
    ax[2].legend(["normal dist. fit ($\mu=${0:,.2%}, $\sigma=${1:,.2%})".format(mu, s)])

def plot_comparison(ts,ts_baseline,params):
    # plt.figure()
    fig, ax = plt.subplots()
    lower_bound, upper_bound = get_confidence_interval(ts,p=params['ci'])
    lower_bound_bs, upper_bound_bs = get_confidence_interval(ts_baseline,p=params['ci'])

    
    ax.fill_between(np.arange(ts.shape[1]), lower_bound, upper_bound, color='green', alpha=0.7,zorder=2,label='portfolio CI')
    ax.fill_between(np.arange(ts.shape[1]), lower_bound_bs, upper_bound_bs, color='gray', alpha=0.7,zorder=1,label='baseline CI')

    ax.axhline(0, color='black', lw=1,linestyle=':')
    ax.axhline(params['starting_price'], color='grey', lw=1,linestyle=':')
    

    ax.legend(bbox_to_anchor=(1.1, 1.05))
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])
    plt.show()

    return fig


def plot_prices_distribution(ts,params):
    '''
    Plot distribution of prices
    '''
    pass