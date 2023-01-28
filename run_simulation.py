import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
from mc import series_gen , portfolio , utils


from scipy.stats import norm
def get_confidence_interval(ts):
    mean = ts.mean(axis=0)
    std = ts.std(axis=0)
    z = norm.ppf(0.975)
    lower_bound = mean - z * std 
    upper_bound = mean + z * std
    return lower_bound, upper_bound


def plot_time_series(ts,params, file_name):
    plt.figure()
    for i in range(ts.shape[0]):
        plt.plot(ts[i,:],alpha = params['plot']['alpha'],zorder =1
        )
    lower_bound, upper_bound = get_confidence_interval(ts)
    
    # lower_bound = np.percentile(ts[:,:], params['plot']['lb'])
    # upper_bound = np.percentile(ts[:,:], params['plot']['ub'])

    plt.fill_between(np.arange(ts.shape[1]), lower_bound, upper_bound, color='gray', alpha=0.7,zorder=2)

    plt.xlabel('time')
    plt.ylabel('price')
    plt.title(params['title'])
    plt.savefig(file_name)
    plt.show()




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
    config = utils.parse_config(args.config)
    return config

def main():
    #create an env, read config
    env = utils.Env()
    config = parse_config()
    print('starting simulations...\nresults will be saved to: ',env.SIM_FOLDER)

    #Generate asset time series  
    sim_res = series_gen.generate_time_series(config.N, config.T,current_price=config.current_price
                    , return_func = series_gen.random_return, params=config.return_function_params, )

    

 

    #run the strategy
    allocated_capital= portfolio.run_one_asset_rebalance_portfolio(time_series=sim_res, 
                           percent_allocated= config.percent_allocated, 
                           threshold= config.rebalance_threshold,
                           k= config.max_rebalances)


    #calculate summary statistics
    run_summary =  (portfolio.ReturnsCalculator(allocated_capital)
                    .calculate_returns()
                    .calculate_stats()
                    )


    # plot data
    params = dict(title= 'MCS: paras:'+str(config.return_function_params)
            ,plot=dict(alpha =0.5,lb=2.5,ub=97.5)
            )
    plot_time_series(sim_res
                    ,params = params
    ,file_name= env.PLOT_TS
    )

    # plot_time_series(allocated_capital
    #                 ,params = dict(title= 'Portfolio')
    # ,file_name= env.PLOT_PORTFOLIO
    # )

    #save data
    save_data(env,sim_res,allocated_capital)
    portfolio.save_stats_to_csv(run_summary,env.STATS_CSV)
    utils.save_config_to_csv(config,env.CONFIG_CSV)



if __name__ == "__main__":
    main()