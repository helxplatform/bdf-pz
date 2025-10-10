import logging
import json
import os
import requests
from requests.exceptions import RequestException
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence, Union, Any, Callable

from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from openai import AuthenticationError as OpenAIAuthenticationError, APIError, APIConnectionError, RateLimitError, OpenAIError, BadRequestError
from transformers import AutoTokenizer

from archytas.models.base import BaseArchytasModel
from archytas.message_schemas import ToolUseRequest
from archytas.exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

logger = logging.getLogger(__name__)

""" ChatOpenAI enforces an API key is set, and an empty string is not accepted. """
DEFERRED_TOKEN_VALUE = "***deferred***"

class ChatVLLM(ChatOpenAI):
    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Optional[
            Sequence[Union[dict[str, Any], type, Callable, BaseTool]]
        ] = None,
    ) -> int:
        return BaseChatModel.get_num_tokens_from_messages(self, messages, tools)

class VLLMModel(BaseArchytasModel):
    api_key: Optional[str] = None

    def __init__(self, config, **kwargs):
        # ModelConfig requires a `model_name: str` field, but this isn't really a requirement
        # for using vLLM, since we will use whatever model is available if not specified.
        if isinstance(config, dict) and "model_name" not in config:
            config["model_name"] = ""

        super().__init__(config, **kwargs)

    @property
    def DEFAULT_MODEL(self) -> str | None:
        if getattr(self, "vllm_models", None):
            return self.vllm_models[0]["id"]
        return None
    
    @property
    def model_name(self) -> str | None:
        # NOTE: Can't use the super getter since it assumes that self._model is defined on the instance.
        _model = getattr(self, "_model", None)
        lc_model_name = getattr(_model, "model", None)
        if isinstance(lc_model_name, str):
            return lc_model_name

        name = getattr(self.config, "model_name", None)
        if not name:
            name = getattr(self, "DEFAULT_MODEL", None)

        # In case the model name has been specified in the LiteLLM vLLM formats.
        if name:
            # Don't need to perform this step if the name comes from _model since it has already been done.
            name = name.removeprefix("hosted_vllm/").removeprefix("vllm/")

        return name
    
    # NOTE: It would be somewhat easy to add support for offline vLLM usage if needed.
    # It's not been implemented here since it has been deprecated by LiteLLM/palimpzest.
    # @property
    # def is_hosted(self) -> bool:
    #     return self.config.get("model_name", self.DEFAULT_MODEL or "").startswith("hosted_vllm")

    def auth(self, **kwargs) -> None:
        """ Load API key, if present. """
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        else:
            auth_token = getattr(self.config, "api_key", None)
        if not auth_token:
            auth_token = os.environ.get("VLLM_API_KEY", os.environ.get("HOSTED_VLLM_API_KEY"))
        if not auth_token:
            # If there is no api key found, we still need to set something or ChatOpenAI will complain about auth. 
            auth_token = DEFERRED_TOKEN_VALUE
            
        # If an API key is not set, assume the vLLM instance does not require auth and use an empty string.
        self.api_key = auth_token
        
        """ Load API base URL of the vLLM instance. """
        base_url = None
        if 'base_url' in kwargs:
            base_url = kwargs['base_url']
        else:
            base_url = getattr(self.config, "base_url", None)
        
        if not base_url:
            base_url = os.environ.get("VLLM_API_BASE", os.environ.get("HOSTED_VLLM_API_BASE"))
        
        if not base_url:
            # If still not defined, all possible sources where the API base could be specified have been exhausted.
            raise AuthenticationError("VLLM API base URL not provided. You may specify it through the model's constructor or config using `base_url` or by setting either `VLLM_API_BASE`/`HOSTED_VLLM_API_BASE`.")
        
        if not base_url.endswith("/"):
            base_url += "/"
        
        self.base_url = base_url

    """
    def load_available_models(self) -> None:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.vllm_models = [model.to_dict() for model in client.models.list()]
        if len(self.vllm_models) == 0:
            raise RuntimeError(f"No models are available under the vLLM instance at { self.base_url }.")
    """

    def load_available_models(self) -> None:
        # Get available models
        try:
            res = requests.get(f"{ self.base_url }models", headers={
                "Authorization": f"Bearer { self.api_key }",
                "Content-Type": "application/json"
            })
            res.raise_for_status()
        except RequestException as e:
            if e.response:
                raise RequestException(f"Failed to retrieve models from vLLM instance at { self.base_url }models. Please ensure the vLLM server is running and accessible.\nStatus: {e.response.status_code}\nBody: {e.response.text}")
            else:
                raise RequestException(f"Failed to retrieve models from vLLM instance at { self.base_url }models. Please ensure the vLLM server is running and accessible. Request failed to send: { e }")
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve models from vLLM instance at { self.base_url }models. Please ensure the vLLM server is running and accessible. Error: {e}")
        
        self.vllm_models = res.json()["data"]
        if len(self.vllm_models) == 0:
            raise RuntimeError(f"No models are available under the vLLM instance at { self.base_url }.")
        
    def get_vllm_model_spec(self, model_name: str) -> dict:
        if getattr(self, "vllm_models", None) is None:
            raise RuntimeError(
                f"Attempt to load spec for '{ model_name }' before vLLM models have been loaded. "
                "Please ensure that `load_available_models` method has been invoked prior to use."
            )

        for model in self.vllm_models:
            if model["id"] == model_name: return model
        
        raise KeyError(
            f"Model '{ model_name }' not found in loaded vLLM models. "
            f"Models available for use: { ', '.join([m['id'] for m in self.vllm_models]) }"
        )

    def initialize_model(self, **kwargs):
        self.load_available_models()

        # Could handle connection errors here.
        return ChatVLLM(
            model=self.model_name,
            api_key=self.api_key,
            openai_api_base=self.base_url
        )

    async def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        # It's possible some vLLM models may not support a temperature kwarg and could cause an error?
        # This is suggested to be the case by OpenAIModel's `ainvoke`.
        # But vLLM does not return which models support a temperature parameter. 

        # We could enforce things like removing `logprobs` kwarg if allow_logprobs not set.
        # But unclear at this point how to handle which permission obj to use, since there may be multiple.
        # model_spec = self.get_vllm_model_spec(self.model_name)
        # perms = model_spec.get("permission", [])

        return await super().ainvoke(input, config=config, stop=stop, **kwargs)
    
    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
        elif isinstance(error, (BadRequestError)) and error.code == "context_length_exceeded":
            raise ContextWindowExceededError(error.body.get('message', None)) from error
        else:
            raise error

    @lru_cache()
    def contextsize(self, model_name = None):
        if model_name is None:
            model_name = self.model_name
        
        model_spec = self.get_vllm_model_spec(model_name)
        try:
            return model_spec["max_model_len"]
        except KeyError:
            logger.warning(f"Could not load context size for '{ self.model_name }' from vLLM spec.")
            logger.debug(json.dumps(model_spec, indent=2))
            return None