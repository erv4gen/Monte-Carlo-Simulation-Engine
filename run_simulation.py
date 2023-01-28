
import imp
import os,  pickle
import matplotlib.pyplot as plt
from collections import namedtuple

import datetime as dt
from mc import series_gen , portfolio

def plot_time_series(ts,params, file_name):
    plt.figure()
    for i in range(ts.shape[0]):
        plt.plot(ts[i,:]#, label='time series {}'.format(i)
        )
    # plt.legend()
    plt.xlabel('time')
    plt.ylabel('price')
    plt.title(params['title'])
    plt.savefig(file_name)
    plt.show()




def save_to_pickle(arr, file_path: str = None):
    with open(file_path, 'wb') as f:
        pickle.dump(arr, f)


def save_data(env,sim_res,allocated_capital):
    save_to_pickle(sim_res,env.TS_FILE_PATH)
    save_to_pickle(allocated_capital,env.TS_PORTFLO_FILE_PATH)


def make_env():
    timestp_ = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    TS_FILE_NAME = 'ts_sims.pkl'
    TS_PORTFLO_FILE_NAME = 'ts_sims.pkl'
    PLOT_FILE_NAME = 'ts_sims.png'
    PLOT_PORTFOLIO_FILE_NAME = 'ts_sims.png'
    SIM_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs',timestp_)

    if not os.path.exists(SIM_FOLDER):
        os.makedirs(SIM_FOLDER)

    print('simulation will be saved to ',SIM_FOLDER)
    
    TS_FILE_PATH = os.path.join(SIM_FOLDER,TS_FILE_NAME)
    TS_PORTFLO_FILE_PATH = os.path.join(SIM_FOLDER,TS_PORTFLO_FILE_NAME)
    
    PLOT_FILE_PATH = os.path.join(SIM_FOLDER,PLOT_FILE_NAME)
    PLOT_PORTFOLIO_FILE_NAME = os.path.join(SIM_FOLDER,PLOT_PORTFOLIO_FILE_NAME)

    

    return namedtuple('RunEnv',['SIM_FOLDER'
                                ,'TS_FILE_PATH'
                                ,'TS_PORTFLO_FILE_PATH'
                                ,'PLOT_FILE_PATH'
                                ,'PLOT_PORTFOLIO_FILE_NAME'
                                ])(SIM_FOLDER,TS_FILE_PATH ,TS_PORTFLO_FILE_PATH, PLOT_FILE_PATH,PLOT_PORTFOLIO_FILE_NAME)
def main():
    
    env = make_env()

    N ,T = 10, 100
    RETURN_FUNC_PARAMS = (0,0.1)

    PERCENT_ALLOCATION = 0.5
    REBALANCE_THRESSHOLD= 0.5
    MAX_REBALANCES = 3


    print('starting simulations...\nresults will be saved to: ',env.SIM_FOLDER)

    #Generate asset simulation 
    sim_res = series_gen.generate_time_series(N, T,current_price=100.
                    , return_func = series_gen.random_return, params=RETURN_FUNC_PARAMS, )

    

 


    allocated_capital= portfolio.rebalance_portfolio(time_series=sim_res, 

                           percent_allocated=PERCENT_ALLOCATION, 
                           threshold= REBALANCE_THRESSHOLD,
                           k= MAX_REBALANCES)

        
    run_summary = portfolio.calculate_return(allocated_capital)

    plot_time_series(sim_res
                    ,params = dict(title= 'MCS: paras:'+str(RETURN_FUNC_PARAMS))
    ,file_name= env.PLOT_FILE_PATH
    )

    plot_time_series(allocated_capital
                    ,params = dict(title= 'Portfolio')
    ,file_name= env.PLOT_PORTFOLIO_FILE_NAME
    )


    save_data(env,sim_res,allocated_capital)
    



if __name__ == "__main__":
    main()