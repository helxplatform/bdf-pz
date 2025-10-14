import logging
import os
import importlib
import requests
from typing import Optional
from openai import OpenAI
from palimpzest.constants import Model as ModelType
from aenum import extend_enum

logger = logging.getLogger(__name__)

VLLM_PZ_MODELS = []

""" NOTE: Once bdf-pz/palimpzest are more stable, monkeypatching should be replaced. """
def monkeypatch_vllm_palimpzest(palimpzest_module) -> None:
    # Ensure appropriate submodules are loaded.
    pz_required_submodules = [".utils.model_helpers", ".constants"]
    for submodule in pz_required_submodules:
        importlib.import_module(submodule, package=palimpzest_module.__name__)

    # Initial `get_models`
    _pz_get_models = palimpzest_module.utils.model_helpers.get_models

    # If `__bdf_patched__` exists on the module's `get_models` function,
    # then it has already been monkeypatched. 
    if getattr(_pz_get_models, "__bdf_patched__", None) == True:
        logger.warning(f"Palimpzest module { palimpzest_module } has already been monkey patched.")
        return

    def get_models_monkeypatch(**kwargs):
        import os
        """
        Palimpzest should already be doing this, as it checks for appropriate environment vars for other LLM providers.
        However, it only enables vLLM models if the `api_base` argument is specified.
        """
        models = _pz_get_models(**kwargs)

        # Strangely, Palimpzest assumes that precanned models like `hosted_vllm/qwen/Qwen1.5-0.5B-Chat` will be available
        # on the user's configured vLLM instance.
        # There's no actual reason to believe these will be available, so they're removed.
        unavailable_vllm_models = [model for model in models if model.is_vllm_model() and model not in VLLM_PZ_MODELS]
        models = [model for model in models if model not in unavailable_vllm_models]

        # vLLM enabled (`api_base` case is handled by original `_pz_get_models`).
        if (
            os.environ.get("HOSTED_VLLM_API_BASE") is not None or
            os.environ.get("VLLM_API_BASE") is not None or
            len(VLLM_PZ_MODELS) > 0
        ):
            # Gather all registered vLLM models, but don't include duplicates (which may exist if `api_base` is specified)
            vllm_models = [model for model in VLLM_PZ_MODELS if model not in models]
            if not kwargs.get("include_embedding", None):
                # remove embedding models
                vllm_models = [model for model in vllm_models if not model.is_embedding_model()]
            models.extend(vllm_models)

        return models

    get_models_monkeypatch.__bdf_patched__ = True
    palimpzest_module.utils.model_helpers.get_models = get_models_monkeypatch

def register_vllm_models_pz(palimpzest_module, models: list[dict]) -> list[ModelType]:
    vllm_models = []
    for model in models:
        # Register the model enum in palimpzest
        raw_model_id = model["id"]
        model_id = f"hosted_vllm/{ raw_model_id }"
        model_name = f"VLLM_{ raw_model_id.split("/")[-1].replace("-", "_").replace(".", "_").upper() }"
        extend_enum(palimpzest_module.constants.Model, model_name, model_id)

        # Register the model cost/performance metrics. This info isn't available from vLLM.
        model_enum = palimpzest_module.constants.Model[model_name]
        palimpzest_module.constants.MODEL_CARDS[model_enum] = {
            "usd_per_input_token": 0,
            "usd_per_output_token": 0,
            "seconds_per_output_token": 0,
            "overall": 0
        }

        vllm_models.append(model_enum)

    return vllm_models

def setup_vllm_palimpzest(
    palimpzest_module # i.e., the top-level import of the palimpzest package.
) -> list[ModelType]:
    monkeypatch_vllm_palimpzest(palimpzest_module)

    try:
        vllm_api_key = os.environ.get("HOSTED_VLLM_API_KEY", os.environ.get("VLLM_API_KEY"))
        vllm_base_url = os.environ["VLLM_API_BASE"]
        if not vllm_base_url.endswith("/"):
            vllm_base_url += "/"
    except KeyError:
        logger.warning("No vLLM URL has been configured; vLLM models will be unavailable.")
        return []
        
    logger.debug("Attempting to fetch available vLLM models")
    # Get available models
    client = OpenAI(api_key=vllm_api_key or "<null>", base_url=vllm_base_url)
    models = [model.to_dict() for model in client.models.list().data]

    pz_models = register_vllm_models_pz(palimpzest_module, models)
    for model in pz_models:
        if model not in VLLM_PZ_MODELS:
            VLLM_PZ_MODELS.append(model)

    return VLLM_PZ_MODELS