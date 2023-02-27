import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
from mc import executor, series_gen , utils , plotting , analysis
import warnings

# plt.figure()
def save_to_pickle(arr, file_path: str = None):
    with open(file_path, 'wb') as f:
        pickle.dump(arr, f)


def save_data(env,sim_res,allocated_capital):
    save_to_pickle(sim_res,env.TS_SIMS)
    save_to_pickle(allocated_capital,env.TS_PORTFLO_SIM)



def parse_config() -> utils.Config:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="path to config file")
    args = parser.parse_args()
    #dict data
    config = utils.read_config(args.config)
    return config

def main():
    #create an env, read config
    env = utils.Env().create_run_env()
    config =  parse_config()
    
    if config.save_logs:
        config = utils.Config(logs_dir = env.LOGS_FOLDER,**config.to_dict())
    
    print('starting simulations...\nresults will be saved to: ',env.SIM_FOLDER,'\nrun parameters:',config)

    #Generate asset time series  
    sim_res = series_gen.generate_time_series(config.return_function_params['N']
                                        , config.return_function_params['T']
                                        ,current_price=config.return_function_params['current_price']
                    , return_func = series_gen.return_functions(config.return_function)
                    , params=config.return_function_params,)

    

 

    #run the strategy
    one_asset_strategy_params = utils.StrategyParams(**config.strategy_function_params)
    allocated_capital= executor.run_one_asset_rebalance_portfolio_v1(time_series=sim_res
                                        ,strategy_params=one_asset_strategy_params
                                        ,config = config
                           )

    #baseline strategy
    baseline_functio_params = utils.StrategyParams()
    baseline_non_allocated= executor.run_one_asset_rebalance_portfolio_v1(time_series=sim_res
                                        ,strategy_params=baseline_functio_params
                                        ,config = config
                           )

    #calculate summary statistics
    run_summary =  (analysis.ReturnsCalculator(allocated_capital,risk_free_rate=config.strategy_function_params['cash_interest'])
                    .calculate_returns()
                    .calculate_stats()
                    )
    baseline_returns =  (analysis.ReturnsCalculator(baseline_non_allocated)
                    .calculate_returns()
                    .calculate_stats()
                    )
    print('simulation stats:\n',run_summary.stats_str)

    # plot data
    prices_plot = plotting.plot_simulations(sim_res
                    ,params = dict(title= 'MCS: paras:'+str(config.return_function_params)
                                ,plot=dict(alpha =0.8)
                                ,ci =config.plot_params['ci'] 
                                ,xlabel='Time, Days'
                                ,ylabel ='Price'
                                )
                                ,show_plot=config.plot_params['show_plot']
                                )

    histigrams_plot = plotting.plot_histogram(sim_res,params = dict(starting_price = config.return_function_params['current_price']
                                )
                                ,show_plot=config.plot_params['show_plot']
                            )


    portfolio_plot = plotting.plot_simulations(run_summary.sim_portfolio
                    ,params = dict(title= 'MCS: paras:'+str(config.return_function_params)
                                ,plot=dict(alpha =0.5)
                                ,ci = config.plot_params['ci'] 
                                ,xlabel='Time, Days'
                                ,ylabel ='Portfolio Value'
                                )
                                ,show_plot=config.plot_params['show_plot']
                                )
    


    comp_plot_parmas = dict(title= config.strategy_function_params['ticker_name']+' Monte Carlo Simulation: Model Portfolio vs Benchmark'
                                ,ci_model_name= str(100* config.plot_params['ci'])+'% Confidence Interval: Model Portfolio'
                                ,ci_benchmark_name= str(100* config.plot_params['ci'])+'% Confidence Interval: Benchmark'
                                ,plot=dict(alpha =0.8)
                                ,ci = config.plot_params['ci'] 
                                ,xlabel='Time, Days'
                                ,ylabel ='Expected Return'
                                ,starting_price = config.return_function_params['current_price']
                                )

    text_box_message =  utils.ComparisonAnnotation(
                                                        sigma=config.return_function_params['sigma']
                                                        ,price_model=config.return_function
                                                        ,n_sims=config.return_function_params['N']
                                                        ,n_steps=config.return_function_params['T']
                                                        ,benchmark=config.strategy_function_params['benchmark_strategy_name'] +' '+config.strategy_function_params['ticker_name']
                                                        ,percent_allocated = config.strategy_function_params['percent_allocated']
                                                        ,rebalance_events =  str(config.strategy_function_params['rebalance_threshold_down']) +'>S>'+str(config.strategy_function_params['rebalance_threshold_up'])
                                                        ,cash_interest = config.strategy_function_params['cash_interest']
                                                        ,staking_rate = config.strategy_function_params['coin_interest']
                                                        ,option_rate = config.strategy_function_params['option_premium']
                                                        ,stats = run_summary.stats_str
                                                        )                       
    comparison_plot_data = plotting.plot_comparison(baseline_returns.sim_portfolio,run_summary.sim_portfolio
                            ,params = comp_plot_parmas
                            ,param_box_message= text_box_message.render_param_str()
                            ,stats_box_message= text_box_message.render_stats_str()
                                                        ,show_plot=config.plot_params['show_plot']
                                                    )
    comp_plot_parmas.update(dict(title= config.strategy_function_params['ticker_name']+' Monte Carlo Simulation: Buy-and-Hold'))
    text_box_message.benchmark = ''
    baseline_only_plot_data = plotting.plot_comparison(baseline_returns.sim_portfolio,ts=None
                            ,params = comp_plot_parmas
                            ,param_box_message= text_box_message.render_param_str()
                            ,stats_box_message= text_box_message.render_stats_str()
                                                        ,show_plot=config.plot_params['show_plot']
                                                    )
    
    

    #save data
    plotting.save_plot(prices_plot,file_name= env.PLOT_TS)
    plotting.save_plot(portfolio_plot,file_name= env.PLOT_PORTFOLIO)

    
    plotting.save_plot(comparison_plot_data,file_name= env.PLOT_COMPARISON)
    plotting.save_plot(baseline_only_plot_data,file_name= env.PLOT_BASELINEONLY)
    plotting.save_plot(histigrams_plot,file_name= env.PLOT_HISTOGRAMS)
    
    save_data(env,sim_res,allocated_capital)
    utils.save_stats_to_csv(run_summary,env.STATS_CSV)
    utils.save_config_to_csv(config,env.CONFIG_CSV)



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()