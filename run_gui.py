from collections import namedtuple
from dataclasses import asdict
from typing import List
import matplotlib , warnings
matplotlib.use('Agg')
import gradio as gr
from mc import utils , engine,series_gen , names , data_source
import os

market_data = data_source.load_market_data(lookback_days=30)


APP_VERSION = os.environ.get('APP_VERSION','no-version-provided')

def hide_plot():
    gr.Plot.update(visible=False)
    
    gr.DataFrame.update(visible=False)
    
def run_mcs_engine(ticker_name:str
                ,return_function:str
                ,investment_amount:float
                ,mu:float
                ,sigma:float
                ,alpha:float
                ,beta:float
                ,delta:float
                ,lambda_:float
                ,N:int
                ,T_str:int
                ,percent_allocated:float
                ,rebalance_threshold:float
                ,cash_interest:float
                ,coin_interest:float
                ,option_every_itervals:int
                ,option_duration:int
                ,show_legend:bool
                ):
    
    #lookup N
    T = utils.TIME_INTERVAL_DICT[T_str]
    config = utils.assemble_conifg(return_function=return_function
                             ,return_function_params = dict(mu=mu,sigma=sigma,alpha=alpha,beta=beta,delta=delta,lambda_=lambda_
                            ,N=N
                            ,T=T
                            ,current_price = market_data[ticker_name].current_price
                            ),
                            strategy_function_params=dict(ticker_name=ticker_name,percent_allocated=percent_allocated
                            ,rebalance_threshold_up= rebalance_threshold +1.
                            ,rebalance_threshold_down=1. -rebalance_threshold
                            ,cash_interest=cash_interest
                            ,coin_interest=coin_interest
                            ,option_every_itervals=option_every_itervals
                            ,option_duration=utils.OPTION_EXPIRATION[option_duration]
                            ,amount_multiple = utils.AMOUNT_DICT[investment_amount] /market_data[ticker_name].current_price
                            ))
                          
    print('starting simulations...\nrun parameters:',asdict(config))
    sim_results = (engine.MCSEngine(config)
                    .run()
                  )
    comparison_plot_data_fig = sim_results.plots.comparison_plot_data_ply.fig
    portfolio_plot_fig  = sim_results.plots.portfolio_plot_ply.fig
    cash_capitalization_plot_fig = sim_results.plots.cash_appreciation_plot_ply.fig
    if not show_legend:
        ax = comparison_plot_data_fig.gca()
        ax.get_legend().remove()
    return (comparison_plot_data_fig
            , portfolio_plot_fig
            , cash_capitalization_plot_fig
            , sim_results.summary.run_summary.stats_df)

with gr.Blocks(title='WAD Simulator') as front_page:
    gr.Markdown(
    f"""
    # WadSet Constructor
    {APP_VERSION}
    """)
    with gr.Row():
        with gr.Column():
            ticker_name = gr.Dropdown(names.market_symbols(), label="Ticker",info='Select ticker')
        with gr.Column():
            gr.Markdown(
                """
                Adjust parameters below based on you risk profile and click `Run Simulation` to estimate metrics
                """)
        with gr.Column():
            pass
    with gr.Row():
        with gr.Column():
            mu = gr.Slider(0.00, 0.99,value=0.0,step=0.001, label="Market Drift",info='How much drift (annualized) we expected in future')
            sigma = gr.Slider(0.01, 0.99,value=0.24,step=0.001, label="Market Volatility",info='How much volatility (annualized) we expected in future')
            
            alpha = gr.Slider(0.01, 20.0,value=0.2444,step=0.0001, label="Alpha GHB",info='')
            beta = gr.Slider(0.001, 1.0,value=0.053,step=0.001, label="Beta GHB",info='')
            delta = gr.Slider(0.0001, 0.0,value=0.0003,step=0.0001, label="Delta GHB",info='')
            lambda_ = gr.Slider(-2.01, 0.0,value=-0.52,step=0.001, label="Lambda GHB",info='')
            
            
            N = gr.Slider(2, 1000,value=50, label="Nunber of Simulations",info='Number of independent tragectories to to generate.')
            percent_allocated = gr.Slider(0.01, 0.99,value=0.5, label="Percent Allocated",info='Percent of cappital to allocate into the asset')
            # T = gr.Slider(365, 36500,value=365, label="T")
            T = gr.Radio(list(utils.TIME_INTERVAL_DICT.keys()),value='1y', label="Investment Horizon", info="The duration of the investment")
            investment_amount = gr.Radio(list(utils.AMOUNT_DICT.keys()),value='$10k', label="Initial Capital")
            return_function = gr.Dropdown(list(series_gen.RETURN_FUNCTIONS.keys()),value='Lognormal Random Walk', label="Return Function",info='What function to use to estimation price trajectories')
            
            
        with gr.Column():            

            rebalance_threshold = gr.Slider(0.01, 0.99,value=0.5, label="Rebalance Threshold,%",info='After what absolute change (up or down) we should rebalance back the portfolio')
            cash_interest = gr.Slider(0.01, 0.99,value=0.04, label="Cash Interest",info='SOFR overnight rate')
            coin_interest = gr.Slider(0.01, 0.99,value=0.05, label="Staking Interest",info='Coin staking rate')
            option_every_itervals = gr.Slider(10, 365,value=30, label="Strangle Every Interval",info='How often to selll options')

            # option_duration = gr.Slider(10, 365,value=25, label="Option Expiration T+",info='what')
            option_duration = gr.Radio(list(utils.OPTION_EXPIRATION.keys()),value='25d', label="Option Expiration T+",info='What expiration to use')
            
            show_legend = gr.Checkbox(label="Show Legend",value=True)
            

        with gr.Column():   
            run_button = gr.Button("Run Simulation")         
            create_wadset = gr.Button("Create WadSet",variant='primary')       

    with gr.Row():
        with gr.Column():
            res_plot = gr.Plot(label="Comparison Plot")
            
        with gr.Column():
            summary_stat = gr.Dataframe(
                                    headers=["Metric", "Value"],
                                    datatype=["str", "str"],
                                    label="Summary Statistics",)
    
    with gr.Row():
        with gr.Column():
            portfolio_plot = gr.Plot(label="Portfolios Plot")
            

        with gr.Column():            
            cash_capitalization_plot = gr.Plot(label="Cash Capitalization")
    
    dep = front_page.load(hide_plot, None,None)
    ticker_name.change(fn=lambda symbol: gr.update(value=market_data[symbol].volatility), inputs=ticker_name, outputs=sigma)

    run_button.click(
        run_mcs_engine,inputs=[ticker_name,return_function,
                               investment_amount,
            mu,
            sigma,
            alpha,
            beta,
            delta,
            lambda_,
            N,
            T,
            percent_allocated,
            rebalance_threshold,
            cash_interest,
            coin_interest,
            option_every_itervals,
            option_duration,
            show_legend]
            ,outputs=[res_plot,portfolio_plot,cash_capitalization_plot,summary_stat],
            )

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        front_page.launch(
                        server_name="0.0.0.0",
                        # auth=("wadset", "wadset"),
                        #   server_port=9085,
                          show_api=False
                          )
    