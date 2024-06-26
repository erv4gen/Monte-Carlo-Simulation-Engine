import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

import json
import base64

import datetime as dt
from mc import utils , engine , overnight

# Your API definition
api_backend = Flask(__name__)
CORS(api_backend)
    
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





@api_backend.route('/wadprice', methods=['GET'])
def run_sofr_api():
    """
    params:
    P_t: price at t-1
    """
    PARAMS = ['p_t','amount_blocked','amount_to_mint','n_to_burn','r_overnight','total_numer_wads']
    try:
        for arg_check in PARAMS:
            assert arg_check in request.args, f'Missing {arg_check} argument'
        now = dt.datetime.now()
        now_str = now.strftime('%Y-%m-%d')
        ts = dt.datetime.timestamp(now)
        
        P_t = float(request.args['p_t'])
        AMOUNT_BLOCKED = float(request.args['amount_blocked'])
        AMT_TOMINT = float(request.args['amount_to_mint'])
        N_TO_BURN = float(request.args['n_to_burn'])
        r = float(request.args['r_overnight'])
        NUMBER_OF_WADS = float(request.args['total_numer_wads'])

        price = overnight.wad_coin_variant3(P_t,AMOUNT_BLOCKED,AMT_TOMINT,N_TO_BURN,r,NUMBER_OF_WADS)
        
        return jsonify({'engine_v': '3'
                            ,'price': price
                        ,'timestamp': ts
                        ,'date':now_str
                        })
    except Exception as e:
        if 'debug' in request.args.keys() and request.args['debug']=='true':
            return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({'error': str(e)})
        

@api_backend.route('/overnight', methods=['GET'])
def run_wad_coin_price():
    try:
        import random
        # current date and time
        now = dt.datetime.now()
        ts = dt.datetime.timestamp(now)

        rate = (5 + random.uniform(0.01,0.05) ) / 100
        rate_overnight = rate/ 365.25
        return jsonify({'market': 'sofr'
                            ,'rate_annual': rate
                            ,'rate_overnight': rate_overnight
                        ,'timestamp': ts
                        })
    except Exception as e:
        if 'debug' in request.args.keys() and request.args['debug']=='true':
            return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({'error': str(e)})        

@api_backend.route('/simulation', methods=['POST'])
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

        cash_appreciation_plot_base64 = img_to_base64(sim_results.plots.cash_appreciation_plot_ply.fig)

        prices_plot_base64 = img_to_base64(sim_results.plots.prices_plot_ply.fig)

        single_portfolio_ts_plot_base64 = img_to_base64(sim_results.plots.single_portfolio_ts_plot_ply.fig)

        


        statistics_dict = sim_results.summary.run_summary.stats

        sample_statistics_dict = sim_results.summary.run_summary.sample_stats

        return jsonify({'simulation_plot': str(comparison_plot_base64)
                        ,"cash_appreciation_plot" : str(cash_appreciation_plot_base64)
                        ,"prices_plot": str(prices_plot_base64)
                        ,'sample_portfolio_plot': str(single_portfolio_ts_plot_base64)
                       ,'summary':statistics_dict
                       ,'sample_portfolio_summary': sample_statistics_dict
                       })

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

    api_backend.run(port=port, debug=False)