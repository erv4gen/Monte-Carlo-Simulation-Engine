
from dataclasses import dataclass
import numpy as np
from . import executor, series_gen , utils , plotting , analysis

@dataclass
class ResultSeries:
    allocated_capital: np.array
    sim_res: np.array
@dataclass
class ResultPlots:
    baseline_only_plot_data:plotting.PlotData
    comparison_plot_data:plotting.PlotData
    portfolio_plot:plotting.PlotData
    histigrams_plot:plotting.PlotData
    prices_plot :plotting.PlotData

@dataclass
class ResultSummary:
    run_summary: analysis.ReturnsCalculator


@dataclass
class SimResults:
    series: ResultSeries
    plots: ResultPlots
    summary: ResultSummary


class MCSEngine:
    def __init__(self,config:utils.Config) -> None:
        self._config = config
    
    def run(self)->SimResults:
        #Generate asset time series  
        sim_res = series_gen.generate_time_series(self._config.return_function_params['N']
                                            , self._config.return_function_params['T']
                                            ,current_price=self._config.return_function_params['current_price']
                        , return_func = series_gen.return_functions(self._config.return_function)
                        , params=self._config.return_function_params,)

        

    

        #run the strategy
        one_asset_strategy_params = utils.StrategyParams(**self._config.strategy_function_params)
        allocated_capital= executor.run_one_asset_rebalance_portfolio_v1(time_series=sim_res
                                            ,strategy_params=one_asset_strategy_params
                                            ,config = self._config
                            )

        #baseline strategy
        baseline_functio_params = utils.StrategyParams()
        baseline_non_allocated= executor.run_one_asset_rebalance_portfolio_v1(time_series=sim_res
                                            ,strategy_params=baseline_functio_params
                                            ,config = self._config
                            )

        #calculate summary statistics
        run_summary =  (analysis.ReturnsCalculator(allocated_capital,risk_free_rate=self._config.strategy_function_params['cash_interest'])
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
                        ,params = dict(title= 'MCS: paras:'+str(self._config.return_function_params)
                                    ,plot=dict(alpha =0.8)
                                    ,ci =self._config.plot_params['ci'] 
                                    ,xlabel='Time, Days'
                                    ,ylabel ='Price'
                                    )
                                    ,show_plot=self._config.plot_params['show_plot']
                                    )

        histigrams_plot = plotting.plot_histogram(sim_res,params = dict(starting_price = self._config.return_function_params['current_price']
                                    )
                                    ,show_plot=self._config.plot_params['show_plot']
                                )


        portfolio_plot = plotting.plot_simulations(run_summary.sim_portfolio
                        ,params = dict(title= 'MCS: paras:'+str(self._config.return_function_params)
                                    ,plot=dict(alpha =0.5)
                                    ,ci = self._config.plot_params['ci'] 
                                    ,xlabel='Time, Days'
                                    ,ylabel ='Portfolio Value'
                                    )
                                    ,show_plot=self._config.plot_params['show_plot']
                                    )
        


        comp_plot_parmas = dict(title= self._config.strategy_function_params['ticker_name']+' Monte Carlo Simulation: Model Portfolio vs Benchmark'
                                    ,ci_model_name= str(100* self._config.plot_params['ci'])+'% Confidence Interval: Model Portfolio'
                                    ,ci_benchmark_name= str(100* self._config.plot_params['ci'])+'% Confidence Interval: Benchmark'
                                    ,plot=dict(alpha =0.8)
                                    ,ci = self._config.plot_params['ci'] 
                                    ,xlabel='Time, Days'
                                    ,ylabel ='Expected Return'
                                    ,starting_price = self._config.return_function_params['current_price']
                                    )

        text_box_message =  utils.ComparisonAnnotation(
                                                            sigma=self._config.return_function_params['sigma']
                                                            ,price_model=self._config.return_function
                                                            ,n_sims=self._config.return_function_params['N']
                                                            ,n_steps=self._config.return_function_params['T']
                                                            ,benchmark=self._config.strategy_function_params['benchmark_strategy_name'] +' '+self._config.strategy_function_params['ticker_name']
                                                            ,percent_allocated = self._config.strategy_function_params['percent_allocated']
                                                            ,rebalance_events =  str(self._config.strategy_function_params['rebalance_threshold_down']) +'>S>'+str(self._config.strategy_function_params['rebalance_threshold_up'])
                                                            ,cash_interest = self._config.strategy_function_params['cash_interest']
                                                            ,staking_rate = self._config.strategy_function_params['coin_interest']
                                                            ,option_range = 'Opt. amount: '+str(self._config.strategy_function_params['option_amount_pct_of_notional'])+';range: '+str(self._config.strategy_function_params['option_straddle_pct_from_strike']) if self._config.strategy_function_params['option_amount_pct_of_notional']>0.0 else ''
                                                            ,stats = run_summary.stats_str
                                                            )                       
        comparison_plot_data = plotting.plot_comparison(baseline_returns.sim_portfolio,run_summary.sim_portfolio
                                ,params = comp_plot_parmas
                                ,param_box_message= text_box_message.render_param_str()
                                ,stats_box_message= text_box_message.render_stats_str()
                                                            ,show_plot=self._config.plot_params['show_plot']
                                                        )
        comp_plot_parmas.update(dict(title= self._config.strategy_function_params['ticker_name']+' Monte Carlo Simulation: Buy-and-Hold'))
        text_box_message.benchmark = ''


        baseline_only_plot_data = plotting.plot_comparison(baseline_returns.sim_portfolio,ts=None
                                ,params = comp_plot_parmas
                                ,param_box_message= text_box_message.render_param_str()
                                ,stats_box_message= text_box_message.render_stats_str()
                                                            ,show_plot=self._config.plot_params['show_plot']
                                                        )



        return SimResults(series=ResultSeries(sim_res=sim_res
                                            ,allocated_capital=allocated_capital)
                        ,summary=ResultSummary(run_summary=run_summary)
                        ,plots=ResultPlots(baseline_only_plot_data=baseline_only_plot_data
                                            ,comparison_plot_data= comparison_plot_data
                                            ,portfolio_plot= portfolio_plot
                                            ,histigrams_plot= histigrams_plot
                                            ,prices_plot = prices_plot
                                            )
                        ,                                            
                        )



