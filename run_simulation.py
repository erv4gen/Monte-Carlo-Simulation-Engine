import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
from mc import series_gen , portfolio , utils


from scipy.stats import norm
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
    plt.figure()
    for i in range(ts.shape[0]):
        plt.plot(ts[i,:],alpha = params['plot']['alpha'],zorder =1
        ,linewidth=0.3
        )
    lower_bound, upper_bound = get_confidence_interval(ts,p=params['ci'])

    plt.fill_between(np.arange(ts.shape[1]), lower_bound, upper_bound, color='gray', alpha=0.7,zorder=2)

    plt.axhline(0, color='black', lw=1)
    


    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])
    plt.title(params['title'])
    plt.show()

    return plt.gcf()


def plot_comparison(ts,ts_baseline,params):
    plt.figure()

    lower_bound, upper_bound = get_confidence_interval(ts,p=params['ci'])
    lower_bound_bs, upper_bound_bs = get_confidence_interval(ts_baseline,p=params['ci'])

    
    plt.fill_between(np.arange(ts.shape[1]), lower_bound, upper_bound, color='green', alpha=0.7,zorder=2,label='portfolio CI')
    plt.fill_between(np.arange(ts.shape[1]), lower_bound_bs, upper_bound_bs, color='gray', alpha=0.7,zorder=1,label='baseline CI')

    plt.axhline(0, color='black', lw=1)
    

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])
    plt.title(params['title'])
    plt.show()

    return plt.gcf()



def save_to_pickle(arr, file_path: str = None):
    with open(file_path, 'wb') as f:
        pickle.dump(arr, f)


def save_data(env,sim_res,allocated_capital):
    save_to_pickle(sim_res,env.TS_SIMS)
    save_to_pickle(allocated_capital,env.TS_PORTFLO_SIM)



def parse_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="path to config file")
    args = parser.parse_args()
    config = utils.read_config(args.config)
    return config

def main():
    #create an env, read config
    env = utils.Env()
    config = parse_config()
    print('starting simulations...\nresults will be saved to: ',env.SIM_FOLDER,'\nrun parameters:',config)

    #Generate asset time series  
    sim_res = series_gen.generate_time_series(config.return_function_params['N']
                                        , config.return_function_params['T']
                                        ,current_price=config.return_function_params['current_price']
                    , return_func = series_gen.return_functions(config.return_function)
                    , params=config.return_function_params,)

    

 

    #run the strategy
    one_asset_strategy_params = utils.StrategyParams(**config.strategy_function_params)
    allocated_capital= portfolio.run_one_asset_rebalance_portfolio(time_series=sim_res
                                        ,strategy_params=one_asset_strategy_params
                           )

    #baseline strategy
    baseline_functio_params = utils.StrategyParams()
    baseline_non_allocated= portfolio.run_one_asset_rebalance_portfolio(time_series=sim_res
                                        ,strategy_params=baseline_functio_params
                           )

    #calculate summary statistics
    run_summary =  (portfolio.ReturnsCalculator(allocated_capital)
                    .calculate_returns()
                    .calculate_stats()
                    )
    baseline_returns =  (portfolio.ReturnsCalculator(baseline_non_allocated)
                    .calculate_returns()
                    .calculate_stats()
                    )
    print('simulation stats:\n',run_summary.stats)

    # plot data
    prices_plot = plot_simulations(sim_res
                    ,params = dict(title= 'MCS: paras:'+str(config.return_function_params)
                                ,plot=dict(alpha =0.8)
                                ,ci = 0.975
                                ,xlabel='Time'
                                ,ylabel ='Price'
                                )
                                )

    portfolio_plot = plot_simulations(run_summary.sim_portfolio
                    ,params = dict(title= 'MCS: paras:'+str(config.return_function_params)
                                ,plot=dict(alpha =0.5)
                                ,ci = 0.975
                                ,xlabel='Time'
                                ,ylabel ='Portfolio'
                                )
                                )
    
    plot_comparison(run_summary.sim_portfolio,baseline_returns.sim_portfolio
    ,params = dict(title= 'Portfolio vs Benchmark:'
                                ,plot=dict(alpha =0.8)
                                ,ci = 0.975
                                ,xlabel='Time'
                                ,ylabel ='Portfolio CI'
                                )
                            )
    

    #save data
    save_plot(prices_plot,file_name= env.PLOT_TS)
    save_plot(portfolio_plot,file_name= env.PLOT_PORTFOLIO)
    save_data(env,sim_res,allocated_capital)
    portfolio.save_stats_to_csv(run_summary,env.STATS_CSV)
    utils.save_config_to_csv(config,env.CONFIG_CSV)



if __name__ == "__main__":
    main()