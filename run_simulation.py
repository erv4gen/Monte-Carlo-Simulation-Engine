import pickle
import matplotlib.pyplot as plt
import argparse

from mc import series_gen , portfolio , utils






def plot_time_series(ts,params, file_name):
    plt.figure()
    for i in range(ts.shape[0]):
        plt.plot(ts[i,:]#, label='time series {}'.format(i)
        )
    # plt.legend()
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
    plot_time_series(sim_res
                    ,params = dict(title= 'MCS: paras:'+str(config.return_function_params))
    ,file_name= env.PLOT_TS
    )

    plot_time_series(allocated_capital
                    ,params = dict(title= 'Portfolio')
    ,file_name= env.PLOT_PORTFOLIO
    )

    #save data
    save_data(env,sim_res,allocated_capital)
    portfolio.save_stats_to_csv(run_summary,env.STATS_CSV)
    



if __name__ == "__main__":
    main()