import matplotlib , warnings
import matplotlib , warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from mc import utils , engine,series_gen

def assemble_conifg(return_function,return_function_params,strategy_function_params):
    config =  utils.parse_config()
    config.return_function = return_function
    config.return_function_params.update(return_function_params)
    config.strategy_function_params.update(strategy_function_params)    
    return config

def hide_plot():
    gr.Plot.update(visible=False)
    
    gr.DataFrame.update(visible=False)
    
def run_mcs_engine(return_function:str
                ,sigma:float
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
    config = assemble_conifg(return_function=return_function
                             ,return_function_params = dict(sigma=sigma
                            ,N=N
                            ,T=T),
                            strategy_function_params=dict(percent_allocated=percent_allocated
                            ,rebalance_threshold_up= rebalance_threshold +1.
                            ,rebalance_threshold_down=1. -rebalance_threshold
                            ,cash_interest=cash_interest
                            ,coin_interest=coin_interest
                            ,option_every_itervals=option_every_itervals
                            ,option_duration=option_duration
                            ))
                          
    sim_results = (engine.MCSEngine(config)
                    .run()
                  )
    comparison_plot_data_fig = sim_results.plots.comparison_plot_data.fig
    portfolio_plot_fig  = sim_results.plots.portfolio_plot.fig
    if not show_legend:
        ax = comparison_plot_data_fig.gca()
        ax.get_legend().remove()
    return comparison_plot_data_fig , portfolio_plot_fig, sim_results.summary.run_summary.stats_df

with gr.Blocks(title='WAD Simulator') as front_page:
    gr.Markdown(
    """
    # WadSet Constructor
    - Adjust parameters below based on you risk profile and click `Run Simulation` to estimate metrics
    """)

    with gr.Row():
        with gr.Column():
            
            sigma = gr.Slider(0.01, 0.99,value=0.24, label="Market Volatility")
            N = gr.Slider(2, 10000,value=10, label="Nunber of Simulations")
            percent_allocated = gr.Slider(0.01, 0.99,value=0.5, label="Percent Allocated")
            # T = gr.Slider(365, 36500,value=365, label="T")
            T = gr.Radio(list(utils.TIME_INTERVAL_DICT.keys()),value='1y', label="Investment Horizon", info="Days")
            return_function = gr.Dropdown(list(series_gen.RETURN_FUNCTIONS.keys()),value='Lognormal Random Walk', label="Return Function")
            
            
        with gr.Column():            

            rebalance_threshold = gr.Slider(0.01, 0.99,value=0.5, label="Rebalance Threshold")
            cash_interest = gr.Slider(0.01, 0.99,value=0.04, label="Cash Interest")
            coin_interest = gr.Slider(0.01, 0.99,value=0.05, label="Staking Interest")
            option_every_itervals = gr.Slider(10, 365,value=30, label="Strangle Every Interval")
            option_duration = gr.Slider(10, 365,value=25, label="Option Expiration T+")
            show_legend = gr.Checkbox(label="Show Legend",value=True)
            

    with gr.Row():
        with gr.Column():   
            run_button = gr.Button("Run Simulation")         
        
        with gr.Column():     
            create_wadset = gr.Button("Create WadSet",variant='primary')       


    res_plot = gr.Plot(label="Comparison Plot")
    with gr.Row():
        with gr.Column():
            summary_stat = gr.Dataframe(
                                    headers=["Metric", "Value"],
                                    datatype=["str", "str"],
                                    label="Summary Statistics",)
        with gr.Column():            
            portfolio_plot = gr.Plot(label="Portfolios Plot")
    
    dep = front_page.load(hide_plot, None,None)
    run_button.click(
        run_mcs_engine,inputs=[return_function,
            sigma,
            N,
            T,
            percent_allocated,
            rebalance_threshold,
            cash_interest,
            coin_interest,
            option_every_itervals,
            option_duration,
            show_legend]
            ,outputs=[res_plot,portfolio_plot,summary_stat],
            )

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        front_page.launch(
                        # server_name="0.0.0.0",
                        # auth=("wadset", "wadset"),
                          show_api=False
                          )
    