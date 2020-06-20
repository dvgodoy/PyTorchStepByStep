from subprocess import Popen


def load_jupyter_server_extension(nbapp):
    """serve the bokeh-app directory with bokeh server"""
    Popen(["tensorboard", "--logdir", "runs", "--port", "6006"])
