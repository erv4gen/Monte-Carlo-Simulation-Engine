from mc import  utils , plotting , engine
import warnings
from dataclasses import asdict

def assemble_input_params()->utils.Config:
    '''
    This function returns the config file which is the single-object working input for the engine
    and the env for the runtime
    '''
    #create an env, read config
    env = utils.Env().create_run_env()
    config =  utils.parse_config(default='config.json')
    
    if config.save_logs:
        config.logs_dir=env.LOGS_FOLDER
    print('starting simulations...\nresults will be saved to: ',env.SIM_FOLDER,'\nrun parameters:',asdict(config))
    utils.config_sanity_check(config)

    return config , env
def main():
    config,env = assemble_input_params()

    sim_results = (engine.MCSEngine(config)
                .run()
                )
    
    

    #save data
    plotting.save_plot(sim_results.plots.prices_plot,file_name= env.PLOT_TS)
    plotting.save_plot(sim_results.plots.prices_plot_ply,file_name= env.PLOT_TS_PLY)
    plotting.save_plot(sim_results.plots.portfolio_plot,file_name= env.PLOT_PORTFOLIO)

    plotting.save_plot(sim_results.plots.single_portfolio_ts_plot_ply,file_name= env.PLOT_SINGLE_PORTFOLIO)

    
    plotting.save_plot(sim_results.plots.comparison_plot_data,file_name= env.PLOT_COMPARISON)
    plotting.save_plot(sim_results.plots.baseline_only_plot_data,file_name= env.PLOT_BASELINEONLY)
    plotting.save_plot(sim_results.plots.histigrams_plot,file_name= env.PLOT_HISTOGRAMS)

    plotting.save_plot(sim_results.plots.cash_appreciation_plot,file_name= env.CASH_APPRECIATION)
    
    plotting.save_plot(sim_results.plots.comparison_plot_data_ply,file_name= env.PLOT_COMPARISON_PLY)
    


    utils.save_data(env,sim_results.series.sim_res
                    ,sim_results.series.allocated_capital)

    utils.save_stats_to_csv(sim_results.summary.run_summary,env.STATS_CSV)
    utils.save_config_to_csv(config,env.CONFIG_CSV)



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()