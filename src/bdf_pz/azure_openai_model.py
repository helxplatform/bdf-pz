import logging
import json
import os
import requests
import httpx
from typing import Optional, Sequence, Union, Any, Callable
from typing_extensions import Self
from functools import cached_property
from pydantic import SecretStr, Field, model_validator
from langchain_core.utils.utils import secret_from_env
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.chat_models._client_utils import _get_default_httpx_client, _get_default_async_httpx_client
from openai import (
    OpenAI, AsyncOpenAI, AuthenticationError as OpenAIAuthenticationError, APIError,
    APIConnectionError, RateLimitError, OpenAIError, BadRequestError
)
from openai._models import FinalRequestOptions
from openai._utils import is_mapping
from transformers import AutoTokenizer
from urllib.parse import urljoin

from archytas.models.base import BaseArchytasModel
from archytas.message_schemas import ToolUseRequest
from archytas.exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

logger = logging.getLogger(__name__)

DEFERRED_VALUE = "***deferred***"

class AzureOpenAIProxyMixin:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        **kwargs
    ):
        # OpenAI will check the wrong environment variables for these.
        if api_key is None:
            # OPENAI_API_KEY will be checked. This is wrong and should throw prior to that point.
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", os.environ.get("AZURE_OPENAI_KEY"))
        if api_key is None:
            raise OpenAIError(
                "The api_key client option must be set either by passing api_key to the client or by setting either the AZURE_OPENAI_API_KEY or AZURE_OPENAI_KEY environment variable"
            )
            
        if base_url is None:
            # OpenAI will check OPENAI_BASE_URL and then default to the public OpenAI API URL if not specified.
            # Should throw prior to that point.
            base_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if base_url is None:
            raise OpenAIError(
                "The base_url client option must be set either by passing base_url to the client or by setting the AZURE_OPENAI_ENDPOINT environment variable"
            )
        
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    def _build_request(
        self,
        options: FinalRequestOptions,
        *,
        retries_taken: int = 0,
    ) -> httpx.Request:
        # This is the only endpoint supported by the proxy currently.
        if options.url.endswith("/chat/completions") and is_mapping(options.json_data):
            """ Transform the request to be compliant with the expected format of the proxy.
            (1) Method changes from POST -> GET (json body discarded, only `messages` and `model` are preserved).
            (2) Set authentication header
            (3) Set query param `request` to body['messages']
            (4) Set modelTypeName to the deployment id
            """

            messages = options.json_data.get("messages")
            model_type_name = options.json_data.get("model")
            
            options.method = "GET"
            options.json_data = None

            if not options.headers:
                options.headers = {}
            options.headers["Authorization"] = f"Bearer { self.api_key }"
            
            options.params["modelTypeName"] = model_type_name
            options.params["request"] = json.dumps(messages)
        else:
            logger.warning(f"Attempting to make request against '{ options.url }' using Azure OpenAI proxy. Only the /chat/completions endpoint is supported.")

        return super()._build_request(options, retries_taken=retries_taken)

# There's not actually any point in using `openai.AzureOpenAI` here since the proxy behaves more similarly to
# the `openai.OpenAI` interface (with many caveats).
class AzureOpenAIProxy(AzureOpenAIProxyMixin, OpenAI):
    pass

class AsyncAzureOpenAIProxy(AzureOpenAIProxyMixin, AsyncOpenAI):
    pass


class ChatAzureOpenAIProxy(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("AZURE_OPENAI_API_KEY", default=None)
    )
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        # Check OPENAI_ORGANIZATION for backwards compatibility.
        self.openai_organization = (
            self.openai_organization
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        self.openai_api_base = self.openai_api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
        client_params: dict = {
            "api_key": (
                self.openai_api_key.get_secret_value() if self.openai_api_key else None
            ),
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if self.openai_proxy:
            raise ValueError("The use of 'openai_proxy' is unsupported.")
        
        if not self.client:
            sync_specific = {
                "http_client": self.http_client
                or _get_default_httpx_client(self.openai_api_base, self.request_timeout)
            }
            self.root_client = AzureOpenAIProxy(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions

        if not self.async_client:
            async_specific = {
                "http_client": self.http_async_client
                or _get_default_async_httpx_client(self.openai_api_base, self.request_timeout)
            }

            self.root_async_client = AsyncAzureOpenAIProxy(**client_params, **async_specific)
            self.async_client = self.root_async_client.chat.completions

        return self
    

class AzureOpenAIProxyModel(BaseArchytasModel):
    def __init__(self, config, **kwargs):
        # ModelConfig requires a `model_name: str` field, but this isn't a requirement for using
        # the Azure OpenAI proxy since we can assume the deployment is an available model.
        if isinstance(config, dict) and "model_name" not in config:
            config["model_name"] = ""

        super().__init__(config, **kwargs)

    @property
    def DEFAULT_MODEL(self) -> str | None:
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    
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

        return name

    def auth(self, **kwargs) -> None:
        """ Load API key. """
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        else:
            auth_token = getattr(self.config, "api_key", None)
        if not auth_token:
            auth_token = os.environ.get("AZURE_OPENAI_API_KEY", os.environ.get("AZURE_OPENAI_KEY"))
        if not auth_token:
            auth_token = DEFERRED_VALUE
            
        self.api_key = auth_token
        
        """ Load API base URL of the proxy server. """
        base_url = None
        if 'base_url' in kwargs:
            base_url = kwargs['base_url']
        else:
            base_url = getattr(self.config, "base_url", None)
        
        if not base_url:
            base_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not base_url:
            # If still not defined, all possible sources where the API base could be specified have been exhausted.
            raise AuthenticationError("Azure OpenAI proxy base URL not provided. You may specify it through the model's constructor or config using `base_url` or by setting `AZURE_OPENAI_ENDPOINT`.")
        
        if not base_url.endswith("/"):
            base_url += "/"
        
        self.base_url = base_url
        
    def initialize_model(self, **kwargs):
        return ChatAzureOpenAIProxy(
            model=self.model_name,
            api_key=self.api_key,
            openai_api_base=self.base_url
        )
    
    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
        elif isinstance(error, (BadRequestError)) and error.code == "context_length_exceeded":
            raise ContextWindowExceededError(error.body.get('message', None)) from error
        else:
            raise error