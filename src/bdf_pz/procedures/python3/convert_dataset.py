schema_name = "{{ schema_name }}"
cardinality_str = "{{cardinality}}"

convert_schema = existing_schemas[schema_name]

cardinality = pz.Cardinality.ONE_TO_MANY if cardinality_str == "one_to_many" else pz.Cardinality.ONE_TO_ONE

#assert dataset exists in scope
assert "dataset" in locals(), "Dataset should be defined in the current scope. Please set the input dataset first."
if cardinality == pz.Cardinality.ONE_TO_MANY:
    dataset = dataset.sem_flat_map(convert_schema)
else:
    dataset = dataset.sem_map(convert_schema)
dataset_revisions.append((dataset, ("convert_dataset", schema_name, cardinality_str)))

dataset