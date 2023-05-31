from run_gui import *
from run_api import *
from multiprocessing import Process
def run_gradio():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        front_page.launch(
            server_name="0.0.0.0",
            auth=("wadset", "wadset"),
            # server_port=9085,
            show_api=False
        )

def run_flask():
    api_backend.run(port=7863, debug=False)

if __name__ == "__main__":
    # gradio_process = Process(target=run_gradio)
    flask_process = Process(target=run_flask)

    # Starting the processes
    # gradio_process.start()
    flask_process.start()

    run_gradio()
    
    # Joining the processes so the main process doesn't exit before they do
    # gradio_process.join()
    # flask_process.join()