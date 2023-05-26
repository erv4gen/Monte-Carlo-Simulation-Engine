import sys
from flask import Flask, request, jsonify
import traceback
import pandas as pd
from mc import utils , engine, series_gen , names , data_source
# Your API definition
app = Flask(__name__)
    
    
@app.route('/simulation', methods=['POST'])
def run_simulation():
    try:
        params_json = request.json

        config = utils.assemble_conifg(return_function=params_json['return_function']
                             ,return_function_params = dict(sigma=params_json['sigma']
                            ,N=params_json['N']
                            ,T=params_json['T']
                            ,current_price = params_json['current_price']
                            ),
                            strategy_function_params=dict(ticker_name=params_json['ticker_name'],percent_allocated=params_json['percent_allocated']
                            ,rebalance_threshold_up= params_json['rebalance_threshold'] +1.
                            ,rebalance_threshold_down=1. -params_json['rebalance_threshold']
                            ,cash_interest=params_json['cash_interest']
                            ,coin_interest=params_json['coin_interest']
                            ,option_every_itervals=params_json['option_every_itervals']
                            ,option_duration=utils.OPTION_EXPIRATION[params_json['option_duration']]
                            ,amount_multiple = utils.AMOUNT_DICT[params_json['investment_amount']] /params_json['current_price']
                            ))
        
        sim_results = (engine.MCSEngine(config)
                    .run()
                  )
        
        return jsonify({'prediction': str(prediction)})

    except:

        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    app.run(port=port, debug=True)