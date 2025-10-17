import logging
import os
import requests
import palimpzest
from palimpzest.utils import model_helpers
from palimpzest.query.generators.generators import OpenAIGenerator
from palimpzest import constants
from importlib import metadata
from typing import Optional
from openai import OpenAI
from aenum import extend_enum
from .azure_openai_model import AzureOpenAIProxy

logger = logging.getLogger(__name__)

PZ_VERSION = metadata.version(palimpzest.__package__)
PZ_VERSION_07 = PZ_VERSION.startswith("0.7")

VLLM_PZ_MODELS = []
GEMINI_PZ_MODELS = []
AZURE_OPENAI_PZ_MODELS = []

"""
NOTE: Once bdf-pz/palimpzest are more stable, basically all of this needs to get replaced/overhauled.
NOTE: Almost the entirety of functionality patched into Palimpzest by these functions is supported in
more recent versions of Palimpzest. Once we are off of v0.7, most of this can be overhauled/removed.
"""

def monkeypatch_palimpzest(
    use_azure_openai_proxy: bool=False
) -> None:
    # If `__bdf_patched__` exists on the module's `get_models` function,
    # then it has already been monkeypatched. 
    if getattr(palimpzest, "__bdf_patched__", None) == True:
        logger.warning(f"Palimpzest has already been monkey patched.")
        return
    
    """ Patch Model methods. """
    if not callable(getattr(constants.Model, "is_vllm_model", None)):
        constants.Model.is_vllm_model = lambda self: "hosted_vllm" in self.value.lower()
    if not callable(getattr(constants.Model, "is_google_model", None)):
        constants.Model.is_google_model = lambda self: "google" in self.value.lower() or "gemini" in self.value.lower()
    if not callable(getattr(constants.Model, "is_vertex_model", None)):
        constants.Model.is_vertex_model = lambda self: "vertex_ai" in self.value.lower()
    if PZ_VERSION_07:
        # This method checks against a hard-coded list in v0.7. It makes more sense to reference
        # the mutable priority list for our purposes, since it gets modified at runtime. 
        constants.Model.is_vision_model = lambda self: self in model_helpers.VISION_MODEL_PRIORITY

    """ Patch `get_models` """
    _pz_get_models = model_helpers.get_models
    def get_models_monkeypatch(**kwargs):
        models = _pz_get_models(**kwargs)

        """ On palimpzest v0.7, there is no support for Gemini here. """
        if PZ_VERSION_07:
            gemini_credentials_path = kwargs.get(
                "gemini_credentials_path",
                os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json")
            )
            if os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY")) is not None or os.path.exists(gemini_credentials_path):
                vertex_models = [model for model in constants.Model if model.is_vertex_model()]
                google_models = [model for model in constants.Model if model.is_google_model()]
                if not kwargs.get("include_embedding", None):
                    vertex_models = [
                        model for model in vertex_models if not model.is_embedding_model()
                    ]
                if kwargs.get("use_vertex", None):
                    models.extend(vertex_models)
                else:
                    models.extend(google_models)

        """ Azure OpenAI proxy won't be detected since it only checks for OPENAI_API_KEY. """
        if os.getenv("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_KEY")) is not None:
            models.extend(AZURE_OPENAI_PZ_MODELS)

        # Strangely, Palimpzest assumes that precanned models like `hosted_vllm/qwen/Qwen1.5-0.5B-Chat` will be available
        # on the user's configured vLLM instance.
        # There's no actual reason to believe these will be available, so they're removed.
        unavailable_vllm_models = [model for model in models if model.is_vllm_model() and model not in VLLM_PZ_MODELS]
        models = [model for model in models if model not in unavailable_vllm_models]

        """
        Palimpzest should already be doing this, as it checks for appropriate environment vars for other LLM providers.
        However, it only enables vLLM models if the `api_base` argument is specified.
        """
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

        # For some reason, palimpzest defaults to the order of this list to choose which model to use
        # in a lot of places.
        order = model_helpers.VISION_MODEL_PRIORITY if kwargs.get("include_vision") else model_helpers.TEXT_MODEL_PRIORITY
        return sorted(models, key=lambda m: order.index(m))
    model_helpers.get_models = get_models_monkeypatch

    if use_azure_openai_proxy:
        # Monkeypatch Palimpzest's OpenAI completion functionality to go through the Azure proxy.
        def get_azure_openai_proxy_client(self, **kwargs):
            # All config should be available in the env.
            return AzureOpenAIProxy(**kwargs)
        OpenAIGenerator._get_client_or_model = get_azure_openai_proxy_client

    palimpzest.__bdf_patched__ = True

def to_variable_notation(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").replace("/", "_").upper()

def register_model_pz(
    model_name: str,
    model_id: str,
    model_card: dict={
        "usd_per_input_token": 0,
        "usd_per_output_token": 0,
        "seconds_per_output_token": 0,
        "overall": 0
    },
    is_text_model: bool=False,
    is_vision_model: bool=False
) -> constants.Model:
    # Register model in Model enum
    extend_enum(constants.Model, model_name, model_id)
    enum = constants.Model[model_name]
    # Register model card
    constants.MODEL_CARDS[enum] = model_card

    if PZ_VERSION_07:
        if is_text_model:
            model_helpers.TEXT_MODEL_PRIORITY.append(enum)
        if is_vision_model:
            model_helpers.VISION_MODEL_PRIORITY.append(enum)

    return enum

def setup_vllm_palimpzest() -> list[constants.Model]:
    pz_models = []
    try:
        vllm_api_key = os.environ.get("HOSTED_VLLM_API_KEY", os.environ.get("VLLM_API_KEY"))
        vllm_base_url = os.environ["VLLM_API_BASE"]
        if not vllm_base_url.endswith("/"):
            vllm_base_url += "/"
    except KeyError:
        logger.info("No vLLM URL has been configured. vLLM models will be unavailable.")
        return []
        
    logger.debug("Attempting to fetch available vLLM models")
    # Get available models
    client = OpenAI(api_key=vllm_api_key or "<null>", base_url=vllm_base_url)
    try:
        models = [model.to_dict() for model in client.models.list().data]
    except Exception as e:
        logger.error(f"Failed to retrieve models from vLLM instance at { vllm_base_url }models. Please ensure the vLLM server is running and accessible.")
        raise e

    for model in models:
        raw_model_id = model["id"]
        model_id = f"hosted_vllm/{ raw_model_id }"
        model_name = "VLLM_" + to_variable_notation(raw_model_id)
        # Assume it's a text model. This is pretty bad but not much to be done in current state.
        pz_models.append(register_model_pz(model_name, model_id, is_text_model=True))

    return pz_models

def setup_gemini_palimpzest() -> list[constants.Model]:
    pz_models = []
    gemini_models = [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
        "gemini/gemini-flash-latest",
        "gemini/gemini-flash-lite-latest"
    ]
    for model_id in gemini_models:
        model_name = to_variable_notation(model_id)
        pz_models.append(register_model_pz(model_name, model_id, is_text_model=True, is_vision_model=True))

    return pz_models

""" Hard overwrite OpenAI models in Palimpzest with Azure proxy. """
def setup_azure_openai_overwrite() -> tuple[
    list[constants.Model],
    bool
]:
    pz_models = []

    try:
        azure_openai_key = os.environ.get("AZURE_OPENAI_KEY")
        azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    except KeyError:
        logger.info("Azure OpenAI proxy has not been configured. Models will be unavailable.")
        return [], False
        
    model_id = azure_openai_deployment
    model_name = to_variable_notation(model_id)
    existing_model = None
    for model in constants.Model:
        if model.name.lower() == model_name.lower():
            existing_model = model
            break

    if existing_model:
        # Update the value of the existing enum entry for the model.
        # This is really terribly bad practice to do, but it's a
        # necessary monkeypatch for the time being.
        orig_value = existing_model.value
        orig_card = constants.MODEL_CARDS[existing_model]
        object.__setattr__(existing_model, "_value_", model_id)
        constants.Model._value2member_map_.pop(orig_value)
        constants.Model._value2member_map_[model_id] = existing_model

        # Update the model card for the new model id
        del constants.MODEL_CARDS[orig_value]
        constants.MODEL_CARDS[model_id] = orig_card

    else:
        logger.debug(f"No existing model found for '{ model_name }'. Registering with Palimpzest...")
        existing_model = register_model_pz(model_name, model_id, is_text_model=True)

    pz_models.append(existing_model)

    # Delete any OpenAI models besides that which are available via the azure proxy.
    # Currently, if the Azure OpenAI proxy is enabled, then OpenAI support is disabled.
    model_helpers.TEXT_MODEL_PRIORITY = [
        m for m in model_helpers.TEXT_MODEL_PRIORITY
        if not m.is_openai_model() or m == existing_model
    ]
    model_helpers.VISION_MODEL_PRIORITY = [
        m for m in model_helpers.VISION_MODEL_PRIORITY
        if not m.is_openai_model() or m == existing_model
    ]

    return pz_models, True

def setup_palimpzest() -> None:
    if not PZ_VERSION_07:
        # There is no Gemini adapter in Palimpzest v0.7.
        GEMINI_PZ_MODELS.extend(setup_gemini_palimpzest())
    VLLM_PZ_MODELS.extend(setup_vllm_palimpzest())
    
    azure_models, use_azure_openai_proxy = setup_azure_openai_overwrite()
    AZURE_OPENAI_PZ_MODELS.extend(azure_models)
    
    monkeypatch_palimpzest(use_azure_openai_proxy=use_azure_openai_proxy)