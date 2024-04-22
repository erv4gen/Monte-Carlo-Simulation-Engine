# Monte Carlo Simulation Engine
![](https://d2rdhxfof4qmbb.cloudfront.net/wp-content/uploads/20180810161839/monaco.jpg)

## First Instalation

1. Install Python envirement
Anaconda env manager is recommended. Can be downladed [here](https://www.anaconda.com/products/distribution)
```bash
conda create --name mcs python=3.8
```

2. Make sure the Python environment is activated:
```bash
conda activate mcs
```
and Python is installed:

```bash
python --version
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create config file
Make sure the `config.json` is available under the project root dir.

One can use the default settings by renaming `default_config.json` to `config.json`
```bash
mv default_config.json config.json
```

sample parametres for the backtest wih GHD distribution:
```json
{
    "data_mode": "backtest",
    "return_function_params": {
        "mu": -0.0001,
        "alpha": 0.2444,
        "beta":0.053,
        "delta":0.0003,
        "lambda_":-0.52,

        "sigma": -99999.0,
        "current_price": 10000,
        "N": 100,
        "T": 1825
    },
    "return_function": "Generalized Hyperbolic",
    "save_logs": false,
    "strategy_function_params" : {
        "amount_multiple":   1.0,
        
        "percent_allocated": 0.5,
        
        "rebalance_threshold_down":  0.8,
        "rebalance_threshold_up":  1.2,

        "max_rebalances": 365,
        "rebalance_every": 365,

        "cash_interest": 0.01,
        "coin_interest": 0.01,

        
        "option_every_itervals":  365 ,
        "option_duration":  360,
        "option_amount_pct_of_notional":  0.3,
        "option_straddle_pct_from_strike":   0.07,
        
        "ticker_name" :  "ETH",
        "all_series_backtest" : false,
        "benchmark_strategy_name": "Buy and Hold"

        
        
                                }
    ,"plot_params": {"show_plot": false
                    ,"ci":0.975
                    }

}

Sample parametres for the log-normal random walk simulation:
```json
{
    "data_mode": "simulation",
    "return_function_params": {
        "mu": 0,
        "sigma": 0.25,
        "current_price": 100,
        "N": 100,
        "T": 365
    },
    "return_function": "Lognormal Random Walk",
    "save_logs": false,
    "strategy_function_params" : {
        "amount_multiple":   1.0,
        
        "percent_allocated": 0.5,
        
        "rebalance_threshold_down":  0.8,
        "rebalance_threshold_up":  1.2,

        "max_rebalances": 365,
        "rebalance_every": 365,

        "cash_interest": 0.04,
        "coin_interest": 0.05,

        
        "option_every_itervals":  30 ,
        "option_duration":  22,
        "option_amount_pct_of_notional":  0.3,
        "option_straddle_pct_from_strike":   0.07,
        
        "ticker_name" :  "ETH",
        "all_series_backtest" : true,
        "benchmark_strategy_name": "Buy and Hold"

        
        
                                }
    ,"plot_params": {"show_plot": false
                    ,"ci":0.975
                    }

}
```

Distribution-specific params:
- `Generalized Hyperbolic Params`:
```json
{"return_function_params": {
    "mu": -0.0001,    # Smaller movements
    "alpha": 0.5,     # Lower variance
    "beta": 0.1,      # Less skewness
    "delta": 0.001,   # Less extreme events
    "lambda": -0.22   # Negative lambda for lighter tails
}}
```

- `Lognormal Random Walk`:
```json
{"return_function_params": {
        "mu": 0, #center
        "sigma": 0.25, #standard deviation
        "current_price": 100,
        "N": 100,
        "T": 365
    }
}
```

For your specific run adjust parameters if needed.


5. Run the simulation 
To run the simulation by executing the Python scripy:

```bash
python run_simulation.py
```
Upon the simulation completion, results will be saved to the respective subfolder under the `data/runs/<RUNID>`, where `RUNID` is a unique id for the simulation with the format: `YYYYMMDDHHMMSS`


6. Analyse Results
Simulation results will be available in the result folder, which includes:
    1. Plots
    2. Run logs
    3. Summary statistics 
    4. Pickled time series data


## UI Frontend 

There's a frontend available that is run on the `gradio` engine. 
To enable the frontend, run:
```sh
python run_gui.py
```
Then nagivate to the `https://localhost:9085`. If the envirement is properly set up, you shuld see the app:
![](assets/Screenshot%202024-04-22%20at%201.07.17 PM.png)

You can run simulation or backtest by pressing the `Run Simulation` button.
Below is an example of the simulation:
![](assets/Screenshot%202024-04-22%20at%201.09.19 PM.png)


## API Backeind 

There's also an API backend avaialbe for the app that is run on the `flask` endinge.
To enable the API endpoint, run:
```sh
python run_api.py 12345
```
Then you can access the API andpoint at:
```sh
curl -d {<params from the config.josn>} -X POST http://localhost:12345/simulation
```


## Docker Container
There's a `Dockerfile` available, that runs both API and GUI at different ports. You can use it to serve the app as an standalone application or as a part of a bigger framework.
