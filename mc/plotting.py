import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib.figure import Figure
from scipy.stats import norm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from typing import NamedTuple
from enum import Enum
import plotly.io as pio


class PlottingEngine(Enum):
    MATPLOTLIB='MATPLOTLIB'
    PLOTLY='PLOTLY'

PLOTLY_FIG_SCALE = 2
PLOTLY_FIG_CONFIG = {
  'toImageButtonOptions': {
    'format': 'png',
    'filename': 'exported_plot',
    'height': 500,
    'width': 700 ,
    'scale':PLOTLY_FIG_SCALE
  }
}


class PlotData(NamedTuple):
    fig: Figure
    objects: list= None
    params: dict= None
    engine: PlottingEngine= PlottingEngine.MATPLOTLIB

def get_confidence_interval(ts,p):
    mean = ts.mean(axis=0)
    std = ts.std(axis=0)
    z = norm.ppf(p)
    upper_bound = mean + z * std

    lower_bound = np.clip(mean - z * std,a_min=0.,a_max=np.inf)
    return lower_bound, upper_bound

def save_plot(plot_data:PlotData,file_name):    
    if plot_data.engine == PlottingEngine.MATPLOTLIB:
        if plot_data.objects is not None:
            plot_data.fig.savefig(file_name
            ,bbox_extra_artists=plot_data.objects
            ,dpi=300
            ,**plot_data.params)
        else:
            plot_data.fig.savefig(file_name,dpi=300)

    elif plot_data.engine == PlottingEngine.PLOTLY:
        pio.write_image(plot_data.fig,file_name, scale=PLOTLY_FIG_SCALE)


def plot_simulations(ts:np.array,params,fill_between=True,zero_line=True,show_plot=True):
    fig, ax = plt.subplots()
    for i in range(ts.shape[0]):
        plt.plot(ts[i,:],alpha = params['plot']['alpha'],zorder =1
        ,linewidth=0.3
        )
    if fill_between:
        lower_bound, upper_bound = get_confidence_interval(ts,p=params['ci'])

        ax.fill_between(np.arange(ts.shape[1]), lower_bound, upper_bound, color='gray', alpha=0.7,zorder=2)
    if zero_line:
        ax.axhline(0, color='black', lw=1,linestyle=':')
    


    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])
    
    if show_plot:
        plt.show()
    
    return PlotData(fig,engine=PlottingEngine.MATPLOTLIB)


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

    return PlotData(fig,engine=PlottingEngine.MATPLOTLIB)

def plot_comparison(ts_baseline,ts=None,params:dict=None,param_box_message:str='',stats_box_message:str='',show_plot=True) -> PlotData:
    # plt.figure()
    fig, ax = plt.subplots(figsize= (8,5)
    )
    fig.tight_layout()
    ts_n  = ts_baseline.shape[1]

    x_offset = 0.20
    

    

    #plot benchmark
    lower_bound_bs, upper_bound_bs = get_confidence_interval(ts_baseline,p=params['ci'])
    ax.fill_between(np.arange( ts_n), lower_bound_bs, upper_bound_bs, color='gray', alpha=0.5,zorder=1,label=params['ci_benchmark_name'])
    
    
    #plot strategy
    if ts is not None:
        lower_bound, upper_bound = get_confidence_interval(ts,p=params['ci'])
        ax.fill_between(np.arange(ts.shape[1]), lower_bound, upper_bound, color='darkcyan', alpha=0.8,zorder=2,label=params['ci_model_name'])
            

    
    x_min = round(ts_n*-x_offset)
    x_max = round( ts_n*(1.03))
    x_t_min = x_min / (abs(x_min)+  ts_n)

    ax.axhline(0,xmin=x_t_min, color='black', lw=1,linestyle=':')
    ax.axhline(params['starting_price'],xmin=x_t_min, color='grey', lw=1,linestyle=':')
    ax.axhline(params['starting_price'] * 2,xmin=x_t_min, color='black', lw=1,linestyle=':')
    

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=False,ncol=2,frameon=False)

    ax.set_xlim(x_min,x_max)
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])
    
    for pad,loc,text_box_message in [(1,params['starting_price'] * 1.9 //1,param_box_message)
                                ,(2,params['starting_price'] // 1.3,stats_box_message)]:
        ax.text(x_min + ( ts_n *x_offset/4) , loc
        ,text_box_message,
            bbox={'facecolor': 'white',
            'alpha': 0.2,
            'pad': pad,
            'boxstyle':'square',
            }
            ,fontsize=6
            ,ha='left', va='top'
            )

    if show_plot:
        plt.show()

    return PlotData(fig=fig , objects=(legend,) , params=dict(bbox_inches='tight'),engine=PlottingEngine.MATPLOTLIB)


def plot_cash_capitalization(cash_df:pd.DataFrame,params,show_plot=True):
    fig, ax = plt.subplots()
    
    cash_df.plot(ax=ax,alpha = params['plot']['alpha'],linewidth=0.3)

    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])
    
    if show_plot:
        plt.show()
    
    return PlotData(fig,engine=PlottingEngine.MATPLOTLIB)

def plot_cash_capitalization_ply(cash_df:pd.DataFrame,params,show_plot=True):    
    fig = go.Figure()
    names = list(zip(cash_df.columns,['darkcyan','grey']))
    for name,color in names:
        fig.add_trace(
        go.Scatter(
            x=cash_df.index,
            y=cash_df[name],
            mode='lines',
            line=dict(
                color=color,
                width=2.0,
            ),
            opacity=params['plot']['alpha'],
            name=name,
        )
    )
    fig.update_xaxes(title=params['xlabel'])
    fig.update_yaxes(title=params['ylabel'])
    fig.update_layout(title=dict(text = params['title'],xanchor='center',x=0.5)
                      ,legend=dict(
            x=0.5,
            y=-0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
                      )

    if show_plot:
        fig.show(config=PLOTLY_FIG_CONFIG)
    return PlotData(fig,engine=PlottingEngine.PLOTLY)

def plot_cash_capitalization(cash_df:pd.DataFrame,params,show_plot=True):
    fig, ax = plt.subplots()
    
    cash_df.plot(ax=ax,alpha = params['plot']['alpha'],linewidth=0.3)

    
    return PlotData(fig,engine=PlottingEngine.PLOTLY)

def plot_simulations_ply(ts, params, show_plot=True):
    fig = go.Figure()

    for i in range(ts.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=np.arange(ts.shape[1]),
                y=ts[i, :],
                mode='lines',
                line=dict(
                    color='blue',
                    width=0.3,
                ),
                opacity=params['plot']['alpha'],
                showlegend=False,
                hoverinfo='none',
                name=f'Simulation {i+1}',
            )
        )

    lower_bound, upper_bound = get_confidence_interval(ts, p=params['ci'])

        # Combine the lower and upper bounds into a single trace with a fill color
    combined_bound = np.concatenate((lower_bound, upper_bound[::-1]), axis=0)
    x_vals = np.arange(ts.shape[1])
    fig.add_trace(
        go.Scatter(
            x=np.concatenate((x_vals, x_vals[::-1])),
            y=combined_bound,
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.7)',
            line=dict(
                color='gray',
                width=0,
            ),
            showlegend=True,
            hoverinfo='none',
            name='Confidence Interval',
        )
    )

    # fig.add_shape(
    #     type='line',
    #     x0=0,
    #     x1=ts.shape[1]-1,
    #     y0=0,
    #     y1=0,
    #     line=dict(
    #         color='black',
    #         width=1,
    #         dash='dot',
    #     ),
    #     name='Zero Line',
    # )

    fig.update_xaxes(title=params['xlabel'])
    fig.update_yaxes(title=params['ylabel'])
    fig.update_layout(title=dict(text = params['title'],xanchor='center',x=0.5)
                      ,xaxis=dict(showgrid=False
                                 ,showline=True
                                 ,linecolor='black'
                                 ), 
                      yaxis=dict(showgrid=False
                                 ,showline=True
                                 ,linecolor='black'),
                    legend=dict(
                                    x=0.5,
                                    y=-0.12,
                                    xanchor='center',
                                    yanchor='top',
                                    bgcolor='rgba(255, 255, 255, 0)',
                                    bordercolor='rgba(255, 255, 255, 0)'
                                ),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)"
                      )

                      
                      
                      

    if show_plot:
        fig.show(config=PLOTLY_FIG_CONFIG)

    return PlotData(fig,engine=PlottingEngine.PLOTLY)





def plot_comparison_ply(ts_baseline, ts=None, params:dict=None, show_plot=True) -> PlotData:
    fig = go.Figure()
    
    ts_n = ts_baseline.shape[1]
    x_vals = np.arange(ts_n)
    x_offset = 0.20

    #plot benchmark
    lower_bound_bs, upper_bound_bs = get_confidence_interval(ts_baseline, p=params['ci'])
    combined_bound_bs = np.concatenate((lower_bound_bs, upper_bound_bs[::-1]), axis=0)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate((x_vals, x_vals[::-1])),
            y=combined_bound_bs,
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.7)',
            line=dict(
                color='gray',
                width=0,
            ),
            name=params['ci_benchmark_name'],
        )
    )

    #plot strategy
    if ts is not None:
        lower_bound, upper_bound = get_confidence_interval(ts, p=params['ci'])
        combined_bound = np.concatenate((lower_bound, upper_bound[::-1]), axis=0)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate((x_vals, x_vals[::-1])),
                y=combined_bound,
                fill='toself',
                fillcolor='rgba(0, 139, 139, 0.7)',
                line=dict(
                    color='darkcyan',
                    width=0,
                ),
                name=params['ci_model_name'],
            )
        )

    # fig.add_shape(
    #     type="line",
    #     x0=0,
    #     y0=0,
    #     x1=ts_n,
    #     y1=0,
    #     line=dict(
    #         color='black',
    #         width=1,
    #         dash="dash"
    #     )
    # )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=params['starting_price'],
        x1=ts_n,
        y1=params['starting_price'],
        line=dict(
            color='black',
            width=1,
            dash="dash"
        )
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=params['starting_price'] * 2,
        x1=ts_n,
        y1=params['starting_price'] * 2,
        line=dict(
            color='black',
            width=1,
            dash="dash"
        )
    )
    
    # fig.add_shape(
    #     type="line",
    #     x0=0,
    #     y0=params['starting_price'] * 2,
    #     x1=ts_n,
    #     y1=params['starting_price'] * 2,
    #     line=dict(
    #         color='black',
    #         width=1,
    #         dash="dash"
    #     )
    # )

    fig.update_layout(
        title=dict(text = params['title'],xanchor='center',x=0.5),
        legend=dict(
            x=0.5,
            y=-0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        xaxis=dict(title=params['xlabel'],
                                    showgrid=False
                                 ,showline=True
                                 ,linecolor='black'
                                 ),
        yaxis=dict(title=params['ylabel']
                                ,showgrid=False
                                 ,showline=True
                                 ,linecolor='black'),

        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    if show_plot:
        fig.show(config=PLOTLY_FIG_CONFIG)
        
    return PlotData(fig,engine=PlottingEngine.PLOTLY)
