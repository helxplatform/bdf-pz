condition = "{{ filter_expression }}"
computed_from = {{ computed_from }}

dataset = dataset.sem_filter(condition, depends_on=computed_from)
dataset_revisions.append((dataset, ("filter_data", condition, computed_from)))

f"Dataset: { repr(dataset) }\nFields: { dataset.schema.schema()['properties'] }"