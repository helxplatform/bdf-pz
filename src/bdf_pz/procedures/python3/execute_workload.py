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
    preferred_models = [model_id.split() for model_id in preferred_models_str.split(",")]
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
    allow_code_synth={{ allow_code_synth }},
    # RAG currently only works when OpenAI has been configured.
    allow_rag_reduction=os.environ.get("OPENAI_API_KEY", "") != "",
    # Mixture of Agents disabled for now. More costly and takes far longer to execute.
    allow_mixtures=False,
    available_models=available_models
)

results = output.run(config)

results_df = results.to_df()
results_df