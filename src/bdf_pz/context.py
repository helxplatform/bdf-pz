# SPDX-FileCopyrightText: 2024-present Brandon Rose <rose.brandon.m@gmail.com>
#
# SPDX-License-Identifier: MIT
from typing import Dict, Any, TYPE_CHECKING
import os
import logging

from beaker_kernel.lib import BeakerContext
from beaker_kernel.lib.utils import action

from .agent import BdfPzAgent

if TYPE_CHECKING:
    from beaker_kernel.kernel import BeakerKernel

logger = logging.getLogger(__name__)

class BdfPzContext(BeakerContext):
    """
    Biomedical Data Fabric Palimpzest Context Class
    """

    compatible_subkernels = ["python3"]
    SLUG = "bdf-pz"

    def __init__(self, beaker_kernel: "BeakerKernel", config: Dict[str, Any]):
        super().__init__(beaker_kernel, BdfPzAgent, config)
        
    async def setup(self, context_info=None, parent_header=None):
        """
        This runs on setup and invokes the `procedures/python3/setup.py` script to 
        configure the environment appropriately.
        """
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        # For now, these are duplicated due to the use of the `HOSTED` prefix in LiteLLM.
        VLLM_API_BASE = os.environ.get("HOSTED_VLLM_API_BASE", os.environ.get("VLLM_API_BASE"))
        VLLM_API_KEY = os.environ.get("HOSTED_VLLM_API_KEY", os.environ.get("VLLM_API_KEY"))
        VLLM_LOAD_AVAILABLE_MODELS_RETRIES = os.environ.get(
            "HOSTED_VLLM_LOAD_AVAILABLE_MODELS_RETRIES",
            os.environ.get("VLLM_LOAD_AVAILABLE_MODELS_RETRIES", 5)
        )
        VLLM_LOAD_AVAILABLE_MODELS_BACKOFF = os.environ.get(
            "HOSTED_VLLM_LOAD_AVAILABLE_MODELS_BACKOFF",
            os.environ.get("VLLM_LOAD_AVAILABLE_MODELS_BACKOFF", 0.25)
        )
        LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
        
        command = "\n".join(
            [
                self.get_code("setup", { k: v for k, v in {
                    "OPENAI_API_KEY": OPENAI_API_KEY,
                    "VLLM_API_BASE": VLLM_API_BASE,
                    "VLLM_API_KEY": VLLM_API_KEY,
                    "VLLM_LOAD_AVAILABLE_MODELS_RETRIES": VLLM_LOAD_AVAILABLE_MODELS_RETRIES,
                    "VLLM_LOAD_AVAILABLE_MODELS_BACKOFF": VLLM_LOAD_AVAILABLE_MODELS_BACKOFF,
                    "LOG_LEVEL": LOG_LEVEL
                }.items() if v is not None })
            ]
        )
        result = await self.execute(command, raise_on_error=False, surpress_messages=False)
        # Note that there is no way to recover the order of stdout and stderr logs respective to each other.
        stdout_list = result.get("stdout_list", [])
        stderr_list = result.get("stderr_list", [])
        error = result.get("error")
        
        if len(stdout_list) > 0:
            # This needs to be displayed at all times, and should go through the logger regardless
            # of log level, as it's important that stderr's header outputs as a warning log.
            _log_level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            print(
                "┌───────────────────────────────────┐\n" +
                "│   bdf-pz `setup.py` STDOUT        │\n" +
                "└───────────────────────────────────┘")
            logger.setLevel(_log_level)
            print(*[f"[{ i }]: " + line for i, line in enumerate(stdout_list, 1)], end="")
            print()
        if len(stderr_list) > 0:
            print(
                "┌───────────────────────────────────┐\n" +
                "│   bdf-pz `setup.py` STDERR        │\n" +
                "└───────────────────────────────────┘")
            print(*[f"[{ i }]: " + line for i, line in enumerate(stderr_list, 1)], end="")
            print()

        
        """ BUG: If you raise an error within a procedure, beaker does not populate the stdout_list/stderr_list fields. """
        if error is not None:
            # Setup could not complete.
            logger.critical(f"bdf-pz procedure `setup.py` encountered an error and failed to complete: { error['ename'] }: { error['evalue'] }")    

    async def auto_context(self):
            return f"""
            You are an assistant helping biomedical researchers users the Palimpzest library to extract references from scientific papers.
            """.strip()

