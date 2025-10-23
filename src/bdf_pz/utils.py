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
    embedding: bool

class ModelSpec(TypedDict):
    id: str
    name: str
    card: ModelCardSpec
    capabilities: ModelCapabilitiesSpec


CUSTOM_MODEL_SPECS_PATH = os.path.join(os.path.dirname(__file__), "custom_models.yaml")
with open(CUSTOM_MODEL_SPECS_PATH, "r") as f:
    CUSTOM_MODEL_SPECS: list[ModelSpec] = yaml.safe_load(f).values()

def get_model_spec_by_id(id: str) -> ModelSpec | None:
    for model_spec in CUSTOM_MODEL_SPECS:
        if model_spec["id"] == id:
            return model_spec
    return None

VLLM_PZ_MODELS = []
OPENAI_PZ_MODELS = []
    
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

        """ Palimpzest assumes that if OPENAI_API_KEY is set, it is going to have every OpenAI model available to it.
        This is a naive assumption--if, e.g., we are running through an azure proxy, only a subset will be usable.
        """
        unavailable_openai_models = [model for model in models if model.is_openai_model() and model not in OPENAI_PZ_MODELS]
        models = [model for model in models if model not in unavailable_openai_models]
            

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

def register_model(model_spec: ModelSpec) -> constants.Model:
    # Register model in Model enum
    model_name = model_spec["name"]
    model_id = model_spec["id"]
    extend_enum(constants.Model, model_name, model_id)
    enum = constants.Model[model_name]
    # Register model card
    constants.MODEL_CARDS[enum] = model_spec["card"]
    # Register model in litellm if no spec exists.
    litellm_model, litellm_provider, _, _ = litellm.get_llm_provider(model_id)
    try:
        litellm.get_model_info(litellm_model, litellm_provider) # throws if not registered
    except:
        # Need to register model or LiteLLM will complain.
        # This could probably be done better, but the custom models spec would need reworking.
        litellm_cost_map = {
            "usd_per_input_token": "input_cost_per_token",
            "usd_per_output_token": "output_cost_per_token",
            "max_input_tokens": "max_input_tokens"
        }
        litellm_capability_map = {
            "reasoning": "supports_reasoning",
            "vision": "supports_vision",
            "audio": "supports_audio_input"
        }
        litellm.register_model({
            model_id: {
                "litellm_provider": litellm_provider,
                "mode": "embedding" if model_spec["capabilities"]["embedding"] else "chat",
                **{ litellm_cost_map.get(k): v for k, v in model_spec["card"].items() if litellm_cost_map.get(k) },
                **{ litellm_capability_map.get(k): v for k, v in model_spec["capabilities"].items() if litellm_capability_map.get(k) }
            }
        })

    return enum

def setup_vllm_palimpzest() -> list[constants.Model]:
    pz_models = []
    try:
        vllm_api_key = os.environ.get("HOSTED_VLLM_API_KEY", os.environ.get("VLLM_API_KEY"))
        vllm_base_url = os.environ.get("HOSTED_VLLM_API_BASE") or os.environ["VLLM_API_BASE"]
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
            pz_models.append(register_model(model_spec))
        else:
            logger.warning(
                f"No model spec exists for the model ID '{ model_id }'. "
                "Please ensure you've added it to the custom_models spec."
            )

    return pz_models

"""
This is important for loading only the models available from the OpenAI server into Palimpzest.
Otherwise, Palimpzest will assume that every OpenAI model is available when the server may actually
be a proxy with only a subset of models offered.
"""
def setup_openai_palimpzest() -> list[constants.Model]:
    pz_models = []
    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        openai_base_url = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL"))
    except KeyError:
        logger.info("No OpenAI API key has been configured. OpenAI models will be unavailable.")
        return []
        
    logger.debug("Attempting to fetch available OpenAI models")
    # Get available models
    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    try:
        models = [model.to_dict() for model in client.models.list().data]
    except Exception as e:
        logger.error(f"Failed to retrieve models from OpenAI using { client.base_url }models. Please ensure that the OpenAI server is running and accessible.")
        raise e
    
    # Register models in palimpzest.
    for model in models:
        model_id = "openai/" + model["id"]
        # The model must either have an exact match against an existing Palimpzest model or it must
        # be registered in CUSTOM_MODEL_SPECS, as we cannot otherwise infer its capabilities.
        pz_model = next((model for model in constants.Model if model.value == model_id), None)

        if pz_model is None:
            # Look for a custom model spec associated with the model id.
            model_spec = get_model_spec_by_id(model_id)
            if model_spec is None:
                logger.warning(
                    f"No existing Palimpzest model was found with ID '{ model_id }' "
                    "and no custom model spec has been configured for it. "
                    f"To enable the model, ensure you've registered its ID in '{ CUSTOM_MODEL_SPECS_PATH }'."
                )
                continue
            pz_model = register_model(model_spec)

        pz_models.append(pz_model)
        
    return pz_models

def setup_palimpzest() -> None:
    try:
        VLLM_PZ_MODELS.extend(setup_vllm_palimpzest())
    except Exception as e:
        logger.error(f"Failed to load vLLM models for Palimpzest.", exc_info=e)
    try:
        OPENAI_PZ_MODELS.extend(setup_openai_palimpzest())
    except Exception as e:
        logger.error(f"Failed to load OpenAI models for Palimpzest.", exc_info=e)

    # gpt-oss doesn't support reasoning_effort: "minimal" which will be used by default if unspecified
    # and cause the request to fail. There doesn't seem to be any better way to tell litellm to not do this.
    # Can't be dropped through litellm_params / additional_drop_params.
    def fix_reasoning_effort(kwargs):
        if "gpt-oss" in kwargs.get("model"):
            complete_input_dict = kwargs.get("additional_args", {}).get("complete_input_dict", {})
            optional_params = kwargs.get("optional_params", {})
            if kwargs.get("reasoning_effort") == "minimal":
                kwargs["reasoning_effort"] = "low"
            if complete_input_dict.get("reasoning_effort") == "minimal":
                complete_input_dict["reasoning_effort"] = "low"
            if optional_params.get("reasoning_effort") == "minimal":
                optional_params["reasoning_effort"] = "low"
    litellm.input_callback = [fix_reasoning_effort]
    
    monkeypatch_palimpzest()