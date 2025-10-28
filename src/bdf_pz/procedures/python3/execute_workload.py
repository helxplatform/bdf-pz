import os

if "dataset" not in locals():
    output = "{{ output_dataset }}"
else:
    output = dataset

assert isinstance(output, pz.Dataset), "Output should be a Dataset object"

policy_method = "{{ policy_method }}"

if policy_method == "min_cost":
    policy = pz.MinCost()
elif policy_method == "max_quality":
    # policy = pz.MaxQuality()
    policy = pz.MinCost()

preferred_models_str = os.environ.get("PZ_PREFERRED_MODELS", "").strip()
available_models = None
if preferred_models_str:
    preferred_models = [model_id for model_id in preferred_models_str.split(",")]
    validated_models = []
    for model_id in preferred_models:
        try:
            validated_models.append(pz.constants.Model(model_id))
        except ValueError:
            logger.warning(
                f"No Palimpzest model is registered with id '{model_id}' "
                f"({ ', '.join([ m.value for m in pz.constants.Model ]) }). "
                "This model will be ignored."
            )
    if len(validated_models) > 0:
        available_models = validated_models
    else:
        logger.warning(
            "No models specified in PZ_PREFERRED_MODELS are registered in Palimpzest. "
            "Defaulting to any available model."
        )

config = pz.QueryProcessorConfig(
    policy=policy,
    cache=False,
    verbose=False,
    progress=False,
    allow_code_synth="{{ allow_code_synth }}".lower() == "true",
    # RAG is currently hardcoded to use Model.TEXT_EMBEDDING_3_SMALL, so this must be disabled if the model is unavailable.
    # (See: https://github.com/mitdbg/palimpzest/blob/1.0.0/src/palimpzest/query/operators/rag.py#L26)
    allow_rag_reduction=(
        pz.constants.Model.TEXT_EMBEDDING_3_SMALL in pz.utils.model_helpers.get_models(include_embedding=True) and
        os.environ.get("PZ_RAG_ENABLED", "false").lower() == "true"
    ),
    # Once fixed in Palimpzest, should change to this. 
    # allow_rag_reduction=any(m.is_embedding_model() for m in pz.utils.model_helpers.get_models(include_embedding=True))
    # Mixture of Agents disabled for now. More costly and takes far longer to execute.
    allow_mixtures=False,
    available_models=available_models
)

results = output.run(config)

results_df = results.to_df()
results_df