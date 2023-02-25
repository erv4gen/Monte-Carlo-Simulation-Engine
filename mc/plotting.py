import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm
import numpy as np
import seaborn as sns
from scipy import stats
from typing import NamedTuple

class PlotData(NamedTuple):
    fig: Figure
    objects: list= None
    params: dict= None

def get_confidence_interval(ts,p):
    mean = ts.mean(axis=0)
    std = ts.std(axis=0)
    z = norm.ppf(p)
    lower_bound = mean - z * std 
    upper_bound = mean + z * std
    return lower_bound, upper_bound

def save_plot(plot_data:PlotData,file_name):    
    if plot_data.objects is not None:
        plot_data.fig.savefig(file_name
        ,bbox_extra_artists=plot_data.objects
        ,**plot_data.params)
    else:
        plot_data.fig.savefig(file_name)


def plot_simulations(ts,params,show_plot=True):
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
    
    if show_plot:
        plt.show()
    
    return PlotData(fig)


def plot_histogram(ts,params,show_plot=True):
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
    ax[0].set_title('Prices')
    ax[0].set_xlabel('Prices')
    ax[0].set_ylabel('Density')
    ax[0].set_xticklabels(['{:,.1f}'.format(x) for x in ax[0].get_xticks()])
    ax[0].legend(["normal dist. fit ($\mu=${0:,.2f}, $\sigma=${1:,.2f})".format(mu, s)])

    # compute the params of the fitted normal curve
    (mu, s) = stats.norm.fit(log_multi_period_rets)
    ax[1].set_title('Simple Returns')
    ax[1].set_xlabel('Returns')
    ax[1].set_ylabel('Density')
    ax[1].set_xticklabels(['{:,.1%}'.format(x) for x in ax[1].get_xticks()])
    ax[1].legend(["normal dist. fit ($\mu=${0:,.2%}, $\sigma=${1:,.2%})".format(mu, s)])

    # compute the params of the fitted normal curve
    (mu, s) = stats.norm.fit(log_multi_period_rets_log)
    ax[2].set_title('Log Returns')
    ax[2].set_xlabel('Returns')
    ax[2].set_ylabel('Density')
    ax[2].set_xticklabels(['{:,.1%}'.format(x) for x in ax[2].get_xticks()])
    ax[2].legend(["normal dist. fit ($\mu=${0:,.2%}, $\sigma=${1:,.2%})".format(mu, s)])
    
    if show_plot:
        plt.show()

    return PlotData(fig)

def plot_comparison(ts,ts_baseline,params,text_box_message,show_plot=True) -> PlotData:
    # plt.figure()
    fig, ax = plt.subplots()
    lower_bound, upper_bound = get_confidence_interval(ts,p=params['ci'])
    lower_bound_bs, upper_bound_bs = get_confidence_interval(ts_baseline,p=params['ci'])

    
    ax.fill_between(np.arange(ts.shape[1]), lower_bound, upper_bound, color='darkcyan', alpha=0.8,zorder=2,label='Confidence Interval: Model Portfolio')
    ax.fill_between(np.arange(ts.shape[1]), lower_bound_bs, upper_bound_bs, color='gray', alpha=0.5,zorder=1,label='Confidence Interval: Benchmark (asset only)')

    ax.axhline(0, color='black', lw=1,linestyle=':')
    ax.axhline(params['starting_price'], color='grey', lw=1,linestyle=':')
    

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=False,ncol=2,frameon=False)

    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])
    ax.text(10, params['starting_price'] // 4,text_box_message,
        bbox={'facecolor': 'white',
         'alpha': 0.5,
         'pad': 10,}
         ,fontsize=6
         )

    if show_plot:
        plt.show()

    return PlotData(fig , (legend,) , dict(bbox_inches='tight'))
