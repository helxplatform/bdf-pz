import logging
import os
import requests
import yaml
import palimpzest
import litellm
from palimpzest.utils import model_helpers
from palimpzest import constants
from typing import Optional, TypedDict
from openai import OpenAI
from aenum import extend_enum
from .azure_openai_model import LiteLLMAzureOpenAIProxy 

logger = logging.getLogger(__name__)

class ModelCardSpec(TypedDict):
    usd_per_input_token: int
    usd_per_output_token: int
    seconds_per_output_token: int
    overall: int

class ModelCapabilitiesSpec(TypedDict):
    reasoning: bool
    text: bool
    vision: bool
    audio: bool

class ModelSpec(TypedDict):
    id: str
    name: str
    card: ModelCardSpec
    capabilities: ModelCapabilitiesSpec


with open(os.path.join(os.path.dirname(__file__), "custom_models.yaml"), "r") as f:
    CUSTOM_MODEL_SPECS: list[ModelSpec] = yaml.safe_load(f).values()

def get_model_spec_by_id(id: str) -> ModelSpec | None:
    for model_spec in CUSTOM_MODEL_SPECS:
        if model_spec["id"] == id:
            return model_spec
    return None

VLLM_PZ_MODELS = []
AZURE_OPENAI_PZ_MODELS = []
    
def monkeypatch_palimpzest() -> None:
    # If `__bdf_patched__` exists on the module's `get_models` function,
    # then it has already been monkeypatched. 
    if getattr(palimpzest, "__bdf_patched__", None) == True:
        logger.warning(f"Palimpzest has already been monkey patched.")
        return
    
    """ Patch capability methods """
    _pz_is_reasoning_model = constants.Model.is_reasoning_model
    _pz_is_text_model = constants.Model.is_text_model
    _pz_is_vision_model = constants.Model.is_vision_model
    _pz_is_audio_model = constants.Model.is_audio_model
    _pz_is_embedding_model = constants.Model.is_embedding_model
    _pz_is_text_image_multimodal_model = constants.Model.is_text_image_multimodal_model
    _pz_is_text_audio_multimodal_model = constants.Model.is_text_audio_multimodal_model

    def custom_has_capability(model: constants.Model, capability: str) -> bool:
        model_spec = get_model_spec_by_id(model.value)
        if model_spec:
            return model_spec["capabilities"].get(capability, False)
        return False
    
    def is_reasoning_model(model):
        return _pz_is_reasoning_model(model) or custom_has_capability(model, "reasoning")
    def is_text_model(model):
        return _pz_is_text_model(model) or custom_has_capability(model, "text")
    def is_vision_model(model):
        return _pz_is_vision_model(model) or custom_has_capability(model, "vision")
    def is_audio_model(model):
        return _pz_is_audio_model(model) or custom_has_capability(model, "audio")
    def is_embedding_model(model):
        return _pz_is_embedding_model(model) or custom_has_capability(model, "embedding")
    def is_text_image_multimodal_model(model):
        return _pz_is_text_image_multimodal_model(model) or (
            custom_has_capability(model, "text") and
            custom_has_capability(model, "vision")
        )
    def is_text_audio_multimodal_model(model):
        return _pz_is_text_image_multimodal_model(model) or (
            custom_has_capability(model, "text") and
            custom_has_capability(model, "audio")
        )
        
    constants.Model.is_reasoning_model = is_reasoning_model
    constants.Model.is_text_model = is_text_model
    constants.Model.is_vision_model = is_vision_model
    constants.Model.is_audio_model = is_audio_model
    constants.Model.is_embedding_model = is_embedding_model
    constants.Model.is_text_image_multimodal_model = is_text_image_multimodal_model
    constants.Model.is_text_audio_multimodal_model = is_text_audio_multimodal_model

    """ Patch `get_models` """
    _pz_get_models = model_helpers.get_models
    def get_models_monkeypatch(
        include_embedding: bool = False,
        use_vertex: bool = True,
        gemini_credentials_path: str | None = None,
        api_base: str | None = None
    ):
        models = _pz_get_models(
            include_embedding=include_embedding,
            use_vertex=use_vertex,
            gemini_credentials_path=gemini_credentials_path,
            api_base=api_base
        )

        """ Azure OpenAI proxy won't be detected since it only checks for OPENAI_API_KEY. """
        if os.getenv("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_KEY")) is not None:
            models.extend([
                m for m in AZURE_OPENAI_PZ_MODELS
                if include_embedding or not m.is_embedding_model()
            ])

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
            models.extend([
                m for m in VLLM_PZ_MODELS
                if m not in models and (include_embedding or not m.is_embedding_model())
            ])

        return models
    
    model_helpers.get_models = get_models_monkeypatch

    setattr(palimpzest, "__bdf_patched__", True)

def register_model_pz(model_spec: ModelSpec) -> constants.Model:
    # Register model in Model enum
    model_name = model_spec["name"]
    model_id = model_spec["id"]
    extend_enum(constants.Model, model_name, model_id)
    enum = constants.Model[model_name]
    # Register model card
    constants.MODEL_CARDS[enum] = model_spec["card"]

    return enum

def setup_vllm_palimpzest() -> list[constants.Model]:
    pz_models = []
    try:
        vllm_api_key = os.environ.get("HOSTED_VLLM_API_KEY", os.environ.get("VLLM_API_KEY"))
        vllm_base_url = os.environ.get("HOSTED_VLLM_API_BASE") or os.environ["VLLM_API_BASE"]
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
        model_spec = get_model_spec_by_id(model_id)
        if model_spec:
            # Assume it's a text model. This is pretty bad but not much to be done in current state.
            pz_models.append(register_model_pz(model_spec))
        else:
            logger.warning(
                f"No model spec exists for the model ID '{ model_id }'. "
                "Please ensure you've added it to the custom_models spec."
            )

    return pz_models

""" Hard overwrite OpenAI models in Palimpzest with Azure proxy. """
def setup_azure_openai_palimpzest() -> list[constants.Model]:
    try:
        azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        azure_openai_key = os.environ.get("AZURE_OPENAI_KEY") or os.environ["AZURE_OPENAI_API_KEY"]
    except KeyError:
        logger.info("Azure OpenAI proxy has not been configured. Models will be unavailable.")
        return []
        
    # Currently there's no way to detect what models are available on the Azure OpenAI proxy
    # besides the deployment, which is guaranteed to exist.
    model_id = "azure-openai-proxy/" + azure_openai_deployment
    model_spec = get_model_spec_by_id(model_id)
    if model_spec is None:
        logger.warning(
            f"No model spec exists for the model ID '{ model_id }'. "
            "Please ensure you've added it to the custom_models spec."
        )
        return []

    pz_model = register_model_pz(model_spec)

    # Register handler with LiteLLM
    azure_openai_proxy_handler = LiteLLMAzureOpenAIProxy()
    litellm.custom_provider_map.append({
        "provider": "azure-openai-proxy",
        "custom_handler": azure_openai_proxy_handler
    })

    # `litellm.completion` completely ignores custom LLM providers whenever
    # the model name corresponds to an OpenAI model... (https://github.com/BerriAI/litellm/issues/14755)
    # So until they fix this, the easiest internal fix is to disable OpenAI entirely in LiteLLM.
    # The consequence of this is that when the azure proxy is enabled, native OpenAI support is disabled.
    litellm.open_ai_chat_completion_models = []

    # Until we have an actual OpenAI-compliant proxy, drop unsupported OpenAI params.
    # The current Azure OpenAI proxy we have running supports no params.
    litellm.drop_params = True

    return [pz_model]

def setup_palimpzest() -> None:
    VLLM_PZ_MODELS.extend(setup_vllm_palimpzest())
    AZURE_OPENAI_PZ_MODELS.extend(setup_azure_openai_palimpzest())
    
    monkeypatch_palimpzest()