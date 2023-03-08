import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from mc import utils , engine

def assemble_conifg(**kwargs):
    config =  utils.parse_config()

    for arg, val in kwargs.items():
        setattr(config,arg,val)
    return config
def run_mcs_engine(return_function:str
                ,sigma:float
                ,N:int
                ,T:int
                ,percent_allocated:float
                ,rebalance_threshold:float
                ,cash_interest:float
                ,coin_interest:float
                ,option_every_itervals:int
                ,option_duration:int
                ,show_legend:bool
                
                ):
    
    config = assemble_conifg(return_function=return_function
                            ,sigma=sigma
                            ,N=N
                            ,T=T
                            ,percent_allocated=percent_allocated
                            ,rebalance_threshold_up= rebalance_threshold +1.
                            ,rebalance_threshold_down=1. -rebalance_threshold
                            ,cash_interest=cash_interest
                            ,coin_interest=coin_interest
                            ,option_every_itervals=option_every_itervals
                            ,option_duration=option_duration
                            )
                          
    sim_results = (engine.MCSEngine(config)
                    .run()
                  )
    comparison_plot_data_fig = sim_results.plots.comparison_plot_data.fig
    if not show_legend:
        ax = comparison_plot_data_fig.gca()
        ax.get_legend().remove()
    return comparison_plot_data_fig

def assemble_conifg(**kwargs):
    config =  utils.parse_config()

with gr.Blocks(title='WAD Simulator') as demo:
    with gr.Row():
        with gr.Column():
            return_function = gr.Dropdown(['log_normal_return', 'random_return'],value='log_normal_return', label="Return Function")
            sigma = gr.Slider(0.01, 0.99,value=0.24, label="Sigma")
            N = gr.Slider(2, 10000,value=100, label="N")
            T = gr.Slider(365, 36500,value=365, label="T")
            percent_allocated = gr.Slider(0.01, 0.99,value=0.5, label="Percent Allocated")

            run_button = gr.Button("Run Simulation")
            
        with gr.Column():
            rebalance_threshold = gr.Slider(0.01, 0.99,value=0.5, label="Rebalance Threshold")
            cash_interest = gr.Slider(0.01, 0.99,value=0.04, label="Cash Interest")
            coin_interest = gr.Slider(0.01, 0.99,value=0.05, label="Staking Interest")
            option_every_itervals = gr.Slider(10, 365,value=30, label="Strangle Every Interval")
            option_duration = gr.Slider(10, 365,value=25, label="Option Expiration T+")
            show_legend = gr.Checkbox(label="Show Legend",value=True)

            
    res_plot = gr.Plot(label="Comparison Plot")
    
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
            ,outputs=[res_plot],
            )

demo.launch()
    