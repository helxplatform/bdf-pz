condition = "{{ filter_expression }}"

dataset = dataset.sem_filter(condition)
dataset_revisions.append((dataset, ("filter_data", condition)))

dataset