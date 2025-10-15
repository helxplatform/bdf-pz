import requests
from beaker_kernel.subkernels.python import PythonSubkernel as BeakerPythonSubkernel
from beaker_kernel.lib.config import config

class PythonSubkernel(BeakerPythonSubkernel):
    def cleanup(self):
        """
        For some inexplicable reason, the DELETE call in `BeakerSubkernel`
        uses a timeout of 0.5 which very often causes it to fail for no reason,
        due to an unreasonably short period of time before it must abort the request.
        To circumvent this, perform the call first to ensure it actually goes through.
        """
        res = requests.delete(
            f"{self.context.beaker_kernel.jupyter_server}/api/kernels/{self.jupyter_id}",
            headers={"Authorization": f"token {config.jupyter_token}"}
        )
        if res.status_code == 204:
            self.jupyter_id = None
        
        # Perform the rest of cleanup. The same DELETE will be now be replayed (and it will most
        # likely raise a ReadTimeout). If it does complete, it will return a 404 instead of a 204
        # since the subkernel has already been deleted. That is fine, however.
        try:
            super().cleanup()
        except requests.exceptions.ReadTimeout as e:
            # Don't care...
            pass