# Monte Carlo Simulation Engine
![Monte Carlo](https://upload.wikimedia.org/wikipedia/commons/2/2f/Monaco_Monte_Carlo_1.jpg)

##First Instalation

1. Install Python envirement
Anaconda env manager is recommended. Can be downladed [here](https://www.anaconda.com/products/distribution)

2. Make sure the Python environment is activated:
```bash
conda activate
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

Adjust parameters if needed.

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

