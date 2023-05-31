from run_gui import *
from run_api import *
from multiprocessing import Process
import argparse
def parse_args_port():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=7861, help="port for the api_backend")
    args = parser.parse_args()
    return args.port

def run_gradio():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        front_page.launch(
            server_name="0.0.0.0",
            auth=("wadset", "wadset"),
            # server_port=9085,
            show_api=False
        )

def run_flask(port):
    api_backend.run(host="0.0.0.0",port=port, debug=False)

if __name__ == "__main__":
    
    flask_port = parse_args_port()
    # gradio_process = Process(target=run_gradio)
    flask_process = Process(target=run_flask,args=(flask_port,))

    # Starting the processes
    # gradio_process.start()
    flask_process.start()

    run_gradio()
    
    # Joining the processes so the main process doesn't exit before they do
    # gradio_process.join()
    # flask_process.join()