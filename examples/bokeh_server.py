import atexit
import contextlib
import socket
import subprocess
import time

import bokeh


def localhost_ip():
    return socket.gethostbyname(socket.gethostname())


class BokehServer(object):
    def __init__(self, name="default", port='6766'):
        self.name = name
        self.port = port
        if port is not None:
            raise NotImplementedError("still figuring out ports..")

        self.server = self._launch()

    def _launch(self):
        command = ["bokeh", "serve"]
        if self.port:
            command.extend(['--port', self.port, '--host', localhost_ip()])
        proc = subprocess.Popen(command, stdout=subprocess.PIPE)
        time.sleep(2)
        return proc

    def terminate(self):
        self.server.terminate()

    def kill(self):
        self.server.kill()

    def output_server(self):
        url = 'http://{}:{}/'.format(localhost_ip(), self.port) if self.port else 'default'
        bokeh.plotting.output_server(self.name, url=url)

    def show(self, p):
        bokeh.plotting.reset_output()
        self.output_server()
        bokeh.plotting.show(p)
        raw_input()


@contextlib.contextmanager
def bokeh_server(name="default", port=None):
    """bokeh_server

    Context manager that launches a `bokeh serve` process for
    plotting.

    Parameters
    ----------
    name : str
        Name to pass into `bokeh.plotting.output_server`
    port : str
        Port to launch the server

    >>> with bokeh_server(name='myplot') as server:
    >>>     x = [1, 2, 3, 4, 5]
    >>>     y = [6, 7, 2, 4, 5]
    >>>     p = bkh.figure(title='simple line example', x_axis_label='x', y_axis_label='y')
    >>>     p.line(x, y, legend="temp", line_width=2)
    >>>     server.show(p)
    """
    server = BokehServer(name=name, port=port)
    @atexit.register
    def kill_bokeh_server():
        server.kill()
    yield server
    server.terminate()
