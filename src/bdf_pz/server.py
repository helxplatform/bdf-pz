import os
import sys
from beaker_kernel.service.server import BeakerServerApp as BeakerLabServerApp
from beaker_kernel.service.base import BeakerServerApp

def _jupyter_server_extension_points():
    module_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    module_path = __package__ + "." + module_name
    return [{"module": module_path, "app": PalimpzestLabServerApp}]

class PalimpzestServerApp(BeakerServerApp):
    @property
    def public_url(self):
        return f"http://{self.ip}:{self.port}{self.base_url}".rstrip("/")


class PalimpzestLabServerApp(BeakerLabServerApp):
    serverapp_class = PalimpzestServerApp


if __name__ == "__main__":
    PalimpzestLabServerApp.launch_instance()