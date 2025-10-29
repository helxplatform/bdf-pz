import os

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

DatasetClass = None
for file in os.listdir(dataset_path):
    _, file_extension = os.path.splitext(file)
    ext = file_extension.lower()
    if ext in extension_map:
        DatasetClass = extension_map[ext]
        break

if DatasetClass is None:
    # If the dataset file extensions are unrecognized, assume they are textual in nature.
    print(
        f"No compatible file types found in directory for '{dataset_name}' ({dataset_path}). "
        "Defaulting to TextFileDataset."
    )
    DatasetClass = pz.TextFileDataset

# Instantiate the chosen dataset class
dataset = DatasetClass(id=dataset_name, path=dataset_path)

dataset