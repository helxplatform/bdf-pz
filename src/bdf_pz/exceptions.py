from typing import Optional

class VLLMNotConfiguredError(Exception):
    def __init__(self):
        message = (
            "No vLLM URL has been configured; vLLM models will be unavailable.\n\n"
            "To use vLLM models, please set the following environment variables:\n"
            "  - VLLM_API_BASE (required)\n"
            "  - VLLM_API_KEY (optional)\n\n"
            "Equivalent variables with a `HOSTED_` prefix are also supported:\n"
            "  - HOSTED_VLLM_API_BASE\n"
            "  - HOSTED_VLLM_API_KEY\n\n"
            "Note: Offline (non-API) usage of vLLM is currently unsupported."
        )
        super().__init__(message)

class VLLMNotReachableError(Exception):
    def __init__(self, message: Optional[str] = None):
        super().__init__(message or "Failed to reach vLLM server.")