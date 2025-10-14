import logging
import palimpzest as pz
import pandas as pd
import os

# formatter = IPython.get_ipython().display_formatter.formatters['text/plain']
# formatter.max_seq_length = 0

#####################
# Setup environment #
#####################
if "{{ OPENAI_API_KEY }}": os.environ["OPENAI_API_KEY"] = "{{ OPENAI_API_KEY }}" 
# For now, these are duplicated due to the usage of the `HOSTED` prefix in LiteLLM (used by Palimpzest).
# The non-prefixed key will be used throughout for consistency as it is more universal.
if "{{ VLLM_API_BASE }}":
    os.environ["VLLM_API_BASE"] = os.environ["HOSTED_VLLM_API_BASE"] = "{{ VLLM_API_BASE }}" 
if "{{ VLLM_API_KEY }}":
    os.environ["VLLM_API_KEY"] = os.environ["HOSTED_VLLM_API_KEY"] = "{{ VLLM_API_KEY }}"
if "{{ LOG_LEVEL }}":
    LOG_LEVEL = os.environ["LOG_LEVEL"] = "{{ LOG_LEVEL }}"

#################
# Setup logging #
#################
"""
NOTE: Unhandled exceptions will terminate the setup process and should be logged rather than escalated unless setup cannot proceed.
NOTE: Procedures execute as "__main__" and therefore will not inherit the package logger, so a handler needs to be configured.
"""
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

############################################################
# Setup locally available models from vLLM with Palimpzest #
############################################################
if os.environ.get("VLLM_API_BASE", "") != "":
    """ Monkeypatch vLLM support and load available vLLM models into palimpzest. """
    from bdf_pz.utils import setup_vllm_palimpzest
    try:
        # `setup_vllm` may throw a few exceptions such as if vLLM environment vars are not configured,
        # if it fails to reach the available model's endpoint, etc.
        VLLM_PZ_MODELS = setup_vllm_palimpzest(pz)
        logger.info("Successfully loaded vLLM into Palimpzest.")
    # Instead of escalating, we should log as an exception if vLLM functionality failed to initialize.
    except Exception as e:
        logger.error("Failed to setup and load vLLM models into Palimpzest.", exc_info=e)

    """ Check if models are available to use from environment """
    ALL_PZ_MODELS_IN_ENV = pz.utils.model_helpers.get_models(include_embedding=True, use_vertex=True)
    if len(ALL_PZ_MODELS_IN_ENV) == 0:
        logger.warning("No models available for use from environment...")
    else:
        logger.info(f"Models available for use in Palimpzest from environment: { ", ".join(ALL_PZ_MODELS_IN_ENV) }")

"""
Misc. preconfigured dataset setup...
"""

# Represents a scientific research paper, which in practice is usually from a PDF file
scientific_paper_schema = [
    {"name": "paper_title", "type": str, "desc": "The title of the paper. This is a natural language title, not a number or letter."},
    {"name": "author", "type": str, "desc": "The name of the first author of the paper"},
    {"name": "abstract", "type": str, "desc": "A short description of the paper contributions and findings"},
]

reference_schema = [
    {"name": "index", "type": int, "desc": "The index of the reference in the paper."},
    {"name": "title", "type": str, "desc": "The title of the paper being cited."},
    {"name": "first_author", "type": str, "desc": "The author of the paper being cited."},
    {"name": "year", "type": int, "desc": "The year in which the cited paper was published."},
]

DATA_PATH = "testdata/"
# print("Setup complete")

registered_datasets = {}
for name in os.listdir(DATA_PATH):
    registered_datasets[name] = os.path.join(DATA_PATH, name)

existing_schemas = {"ScientificPaper":scientific_paper_schema,
                    "Reference":reference_schema}