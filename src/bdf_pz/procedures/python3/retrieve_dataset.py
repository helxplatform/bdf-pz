import os

dataset_name = "{{ dataset_name }}".strip()
dataset_path = registered_datasets[dataset_name]
files = os.listdir(dataset_path)

files