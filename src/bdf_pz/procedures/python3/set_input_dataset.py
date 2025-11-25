import os
from bdf_pz.multimodal_dataset import MultimodalDataset

dataset_name = "{{ dataset_name }}"
dataset_path = registered_datasets[dataset_name]

if not os.path.isdir(dataset_path):
    raise NotADirectoryError(f"Registered path for '{ dataset_name }' is not a valid directory: {dataset_path}")

# Currently, multimodal datasets are unsupported.
extension_map = {
    '.pdf':      pz.PDFFileDataset,
    '.png':      pz.ImageFileDataset,
    '.jpg':      pz.ImageFileDataset,
    '.jpeg':     pz.ImageFileDataset,
    '.gif':      pz.ImageFileDataset,
    '.bmp':      pz.ImageFileDataset,
    '.wav':      pz.AudioFileDataset,
    '.html':     pz.HTMLFileDataset,
    '.htm':      pz.HTMLFileDataset,
    '.xls':      pz.XLSFileDataset,
    '.xlsx':     pz.XLSFileDataset,
    '.txt':      pz.TextFileDataset,
}

found_dataclasses = set()
for file in os.listdir(dataset_path):
    _, file_extension = os.path.splitext(file)
    ext = file_extension.lower()
    if ext in extension_map:
        found_dataclasses.add(extension_map[ext])

DatasetClass = None
if found_dataclasses:
    if len(found_dataclasses) > 1:
        DatasetClass = MultimodalDataset
    else:
        DatasetClass = list(found_dataclasses)[0]
else:
    # If the dataset file extensions are unrecognized, assume they are textual in nature.
    logger.warning(
        f"Warning: No compatible file types found in directory for '{dataset_name}' ({dataset_path}). "
        "Defaulting to TextFileDataset. This will behave erratically if files cannot be interpreted textually."
    )
    DatasetClass = pz.TextFileDataset

# Instantiate the chosen dataset class
dataset = DatasetClass(id=dataset_name, path=dataset_path)
# Track each transformation of the dataset for the purposes of backtracking an action if necessary.
dataset_revisions = [
    (dataset, ("set_input_dataset", dataset_name))
]

f"Dataset: { repr(dataset) }\nFields: { dataset.schema.schema()['properties'] }"