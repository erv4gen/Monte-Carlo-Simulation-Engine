import sys
from flask import Flask, request, jsonify
import traceback
import pandas as pd
import json
import base64
import plotly.graph_objects as go
from mc import utils , engine, series_gen , names , data_source
# Your API definition
app = Flask(__name__)
    
class InvalidInputParameters(Exception):
    pass

def img_to_base64(figure_object):

    # Convert the figure to a static PNG image
    fig_bytes = figure_object.to_image(format="png")

    # Convert bytes to base64 encoded string
    base64_image = base64.b64encode(fig_bytes).decode('utf-8')

    return base64_image

def validate_input(params_json):
    default_config =  utils.parse_config(default='default_config.json')
    requied_keys = ['return_function']  + list(default_config.strategy_function_params)+ list(default_config.return_function_params) + list(default_config.plot_params)

    provided_keys = list(params_json)

    missing_params = list(set(requied_keys)  - set(provided_keys))

    if len(missing_params)>0: raise InvalidInputParameters('Invalid parameters. Missing keys:' + ','.join(missing_params))



@app.route('/simulation', methods=['POST'])
def run_simulation():
    try:
        print('recived params:',dict(request.args.items()))
        params_json = request.json
        validate_input(params_json)
        config = utils.assemble_conifg(return_function=params_json['return_function']
                             ,return_function_params = dict(sigma=params_json['sigma']
                            ,N=params_json['N']
                            ,T=params_json['T']
                            ,current_price = params_json['current_price']
                            ),
                            strategy_function_params=dict(ticker_name=params_json['ticker_name'],percent_allocated=params_json['percent_allocated']
                            ,rebalance_threshold_up= params_json['rebalance_threshold_up']
                            ,rebalance_threshold_down= params_json['rebalance_threshold_up']
                            ,cash_interest=params_json['cash_interest']
                            ,coin_interest=params_json['coin_interest']
                            ,option_every_itervals=params_json['option_every_itervals']
                            ,option_duration=params_json['option_duration']
                            ,amount_multiple = params_json['amount_multiple']
                            ))
        
        sim_results = (engine.MCSEngine(config)
                    .run()
                  )
        
        comparison_plot_base64 = img_to_base64(sim_results.plots.comparison_plot_data_ply.fig)
        statistics_dict = sim_results.summary.run_summary.stats
        return jsonify({'simulation_plot': str(comparison_plot_base64)
                       ,'summary':statistics_dict})

    except Exception as e:
        if 'debug' in request.args.keys() and request.args['debug']=='true':
            return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    app.run(port=port, debug=True)