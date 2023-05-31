
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import executor, series_gen , utils , plotting , analysis

@dataclass
class ResultSeries:
    allocated_capital: np.array
    sim_res: np.array
@dataclass
class ResultPlots:
    baseline_only_plot_data:plotting.PlotData
    comparison_plot_data:plotting.PlotData
    comparison_plot_data_ply:plotting.PlotData
    portfolio_plot:plotting.PlotData
    portfolio_plot_ply:plotting.PlotData
    histigrams_plot:plotting.PlotData
    prices_plot :plotting.PlotData
    prices_plot_ply:plotting.PlotData
    cash_appreciation_plot:plotting.PlotData
    cash_appreciation_plot_ply:plotting.PlotData

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
        baseline_functio_params = utils.StrategyParams(amount_multiple=self._config.strategy_function_params['amount_multiple'])
        baseline_non_allocated= executor.run_one_asset_rebalance_portfolio_v1(time_series=sim_res
                                            ,strategy_params=baseline_functio_params
                                            ,config = self._config
                            )
        
        #cash investemnt comparison
        cash_start = allocated_capital[0,0,utils.ASSET_INDEX['cash']]
        daily_appreceation = series_gen.cash_investment(n=self._config.return_function_params['T']
                                                        ,initial_amount=cash_start
                                                        ,rate=self._config.strategy_function_params['cash_interest']
                                                        ,capitalization_period=1
                                                        )
        mo6_appreceation = series_gen.cash_investment(n=self._config.return_function_params['T']
                                                        ,initial_amount=cash_start
                                                        ,rate=self._config.strategy_function_params['cash_interest']
                                                        ,capitalization_period=179
                                                        )
        cash_interest_comp = pd.DataFrame(np.array([daily_appreceation, mo6_appreceation]).T,columns=['Daily capitalization','Semi-annual capitalization'])


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
        
        portfolio_plot_params =dict(title= 'Trajectories Confidence Interval'
                                    ,plot=dict(alpha =0.5)
                                    ,ci = self._config.plot_params['ci'] 
                                    ,xlabel='Time, Days'
                                    ,ylabel ='Portfolio Value'
                                    )
        
        prices_plot_ply = plotting.plot_simulations_ply(sim_res
                        ,params = portfolio_plot_params
                                    ,show_plot=self._config.plot_params['show_plot']
                                    )

        histigrams_plot = plotting.plot_histogram(sim_res,params = dict(starting_price = self._config.return_function_params['current_price']
                                    )
                                    ,show_plot=self._config.plot_params['show_plot']
                                )

        portfolio_plot = plotting.plot_simulations(run_summary.sim_portfolio
                        ,params = portfolio_plot_params
                                    ,show_plot=self._config.plot_params['show_plot']
                                    )
        
        

        portfolio_plot_ply = plotting.plot_simulations_ply(run_summary.sim_portfolio
                        ,params = portfolio_plot_params
                                    ,show_plot=self._config.plot_params['show_plot']
                                    )



        cash_plot_params =dict(title= 'Cash Capitalization Comparison'
                                    ,plot=dict(alpha =0.5)
                                    ,ci = self._config.plot_params['ci'] 
                                    ,xlabel='Time, Days'
                                    ,ylabel ='Portfolio Value'
                                    )
        
        cash_appreciation_plot = plotting.plot_cash_capitalization(cash_interest_comp
                        ,params = cash_plot_params
                                    ,show_plot=self._config.plot_params['show_plot']
                                    )
        cash_appreciation_plot_ply = plotting.plot_cash_capitalization_ply(cash_interest_comp
                        ,params = cash_plot_params
                                    ,show_plot=self._config.plot_params['show_plot']
                                    )


        comp_plot_parmas = dict(title='Monte Carlo Simulation: Model Portfolio vs Benchmark' #self._config.strategy_function_params['ticker_name']+
                                    ,ci_model_name= str(100* self._config.plot_params['ci'])+'% Confidence Interval: Model Portfolio'
                                    ,ci_benchmark_name= str(100* self._config.plot_params['ci'])+'% Confidence Interval: Benchmark'
                                    ,plot=dict(alpha =0.8)
                                    ,ci = self._config.plot_params['ci'] 
                                    ,xlabel='Time, Days'
                                    ,ylabel ='Expected Return'
                                    ,starting_price = self._config.return_function_params['current_price'] * self._config.strategy_function_params['amount_multiple']
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
        
        
        
        comparison_plot_data_ply = plotting.plot_comparison_ply(baseline_returns.sim_portfolio,run_summary.sim_portfolio
                                ,params = comp_plot_parmas
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
                                            ,comparison_plot_data_ply =comparison_plot_data_ply
                                            ,cash_appreciation_plot = cash_appreciation_plot
                                            ,cash_appreciation_plot_ply = cash_appreciation_plot_ply
                                            ,portfolio_plot= portfolio_plot
                                            ,portfolio_plot_ply=portfolio_plot_ply
                                            ,histigrams_plot= histigrams_plot
                                            ,prices_plot = prices_plot
                                            ,prices_plot_ply=prices_plot_ply
                                            )
                        ,                                            
                        )



