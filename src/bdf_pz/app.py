import os
import json
import logging
from beaker_kernel.lib.app import BeakerApp
from beaker_kernel.lib.autodiscovery import find_mappings

logger = logging.getLogger(__name__)

class PalimpzestApp(BeakerApp):
    slug = "palimpzest"
    name = "PalimpChat"

    pages = {
        "chat": {
            "title": "{app_name}",
            "default": True,
        },
        "notebook": {
            "title": "Palimpzest notebook",
        },
        "dev": {
            "title": "Palimpzest dev interface",
        }
    }

    stylesheet = "global_stylesheet"

    default_context = {
        "slug": "bdf-pz",
        "payload": {},
        "single_context": False,
    }

    assets = {
        "header_logo": {
            "src": "palimpzest-cropped.png",
            "alt": "{app_name} logo",
        },
        "body_logo": {
            "src": "palimpzest-cropped.png",
            "alt": "{app_name} cropped logo",
            "height": "75px"
        },
        "global_stylesheet": {
            "src": "style.css",
        }
    }

    template_bundle = {
        "short_title": "{app_name}",
        "chat_welcome_html": """<div style="display: flex; flex-direction: row; align-items: center; gap: 20px;">
          <img src="{asset:body_logo:src}" alt="{asset:body_logo:alt}" height="{asset:body_logo:height}">
          <p>Hi! I'm your Palimpzest Agent and I can help you do all sorts of information extraction tasks. Your first step is to upload
          some documents or datasets to get started. Then let me know what kind of information you'd like to extract from them.
          Let me know what you'd like to do so I can best assist you!</p>
        </div>""",
    }

    def __init__(self):
        super().__init__()
        self._cleanup_contexts()
        self._cleanup_subkernels()
        
        if os.environ.get("DEBUG", "false") in ["true", "t", "1"]:
            from langchain_core.globals import set_debug
            set_debug(True)

    def _cleanup_contexts(self) -> None:
        # Go through context directories to purge beaker-kernel's default context.
        # I am unsure why there is no straightforward way to disable its installation
        # in the first place.
        context_mappings = find_mappings("contexts")
        for context_path, context_config in context_mappings:
            if context_config["slug"] == "default":
                try:
                    os.remove(context_path)
                    logger.info(f"Purged beaker's default context from '{ context_path }'.")
                except Exception as e:
                    logger.error(f"Failed to delete beaker's default context under '{ context_path }'.", exc_info=e)

    def _cleanup_subkernels(self) -> None:
        # Go through subkernel directories to overwrite beaker's default python subkernel.
        subkernel_mappings = find_mappings("subkernels")

        python3_package_target = "bdf_pz.python_subkernel"
        python3_class_target = "PythonSubkernel"

        for subkernel_path, subkernel_config in subkernel_mappings:
            if subkernel_config["slug"] == "python3":
                try:
                    with open(subkernel_path, "r") as f:
                        conf = json.load(f)
                    with open(subkernel_path, "w") as f:
                        conf["package"] = python3_package_target
                        conf["class_name"] = python3_class_target
                        json.dump(conf, f)
                    logger.info(f"Overwrote beaker's default python3 subkernel from '{ subkernel_path }'.")
                except Exception as e:
                    logger.error(f"Failed to delete beaker's default python3 subkernel under '{ subkernel_path }'.", exc_info=e)