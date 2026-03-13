import os
from bdf_pz.dataset import load_pz_dataset

dataset_name = "{{ dataset_name }}"
dataset_path = registered_datasets[dataset_name]

# Instantiate the chosen dataset class
dataset = load_pz_dataset(dataset_name, dataset_path)
# Track each transformation of the dataset for the purposes of backtracking an action if necessary.
dataset_revisions = [
    (dataset, ("set_input_dataset", dataset_name))
]

f"Dataset: { repr(dataset) }\nFields: { dataset.schema.schema()['properties'] }"