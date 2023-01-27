
import imp
import os,  pickle
import matplotlib.pyplot as plt

import datetime as dt
from mc import series_gen

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


def make_env():
    timestp_ = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    TS_FILE_NAME = 'ts_sims.pkl'
    PLOT_FILE_NAME = 'ts_sims.png'
    SIM_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs',timestp_)

    if not os.path.exists(SIM_FOLDER):
        os.makedirs(SIM_FOLDER)

    print('simulation will be saved to ',SIM_FOLDER)
    
    TS_FILE_PATH = os.path.join(SIM_FOLDER,TS_FILE_NAME)
    PLOT_FILE_PATH = os.path.join(SIM_FOLDER,PLOT_FILE_NAME)

    return TS_FILE_PATH , PLOT_FILE_PATH
def main():
    
    TS_FILE_PATH , PLOT_FILE_PATH = make_env()

    N ,T = 10, 100
    RETURN_FUNC_PARAMS = (0,0.1)


    print('starting simulations...\nresults will be saved to: ',TS_FILE_PATH)
    sim_res = series_gen.generate_time_series(N, T,current_price=100.
                    , return_func = series_gen.random_return, params=RETURN_FUNC_PARAMS, )

    save_to_pickle(sim_res,TS_FILE_PATH)

    print('saving time series plot to ',PLOT_FILE_PATH)

    plot_time_series(sim_res
                    ,params = dict(title= 'MCS: paras:'+str(RETURN_FUNC_PARAMS))
    ,file_name= PLOT_FILE_PATH
    )


if __name__ == "__main__":
    main()