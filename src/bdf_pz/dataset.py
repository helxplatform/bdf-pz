import glob
import os
import logging
import pandas as pd
import palimpzest as pz
from palimpzest.constants import HTML_EXTENSIONS, PDF_EXTENSIONS, XLS_EXTENSIONS, AUDIO_EXTENSIONS, IMAGE_EXTENSIONS
from palimpzest.tools.pdfparser import get_text_from_pdf

logger = logging.getLogger(__file__)

class MultimodalDataset(pz.IterDataset):
    def __init__(self, id: str, path: str):
        self.root_path = path
        
        root_items = [os.path.join(path, i) for i in os.listdir(path)]
        self.is_grouped = any(os.path.isdir(i) for i in root_items)

        # Map modalities to supported extensions
        self.ext_map = {
            "image": IMAGE_EXTENSIONS,
            "audio": AUDIO_EXTENSIONS,
            "text":  [".txt", ".md"] + HTML_EXTENSIONS + PDF_EXTENSIONS,
            "data":  [".csv", ".json", ".yaml", ".xml"] + XLS_EXTENSIONS
        }
        self.modalities_found = {k: False for k in self.ext_map}

        scan_pattern = os.path.join(path, "*/**" if self.is_grouped else "**")
        for filepath in glob.iglob(scan_pattern, recursive=True):
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(filepath)
                ext = ext.lower()
                if ext in self.ext_map["text"]: self.modalities_found["text"] = True
                elif ext in self.ext_map["image"]: self.modalities_found["image"] = True
                elif ext in self.ext_map["audio"]: self.modalities_found["audio"] = True
                elif ext in self.ext_map["data"]: self.modalities_found["data"] = True

        if self.is_grouped:
            self.sources = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        else:
            self.sources = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

        schema = []
        if self.is_grouped:
            schema.append({"name": "entity_id", "type": str, "desc": "The identifier of the input row (directory name)"})
        else:
            schema.append({"name": "filename", "type": str, "desc": "The name of the file"})

        if self.modalities_found["text"]:
            t_type = list[str] if self.is_grouped else str | None
            schema.append({"name": "text_content", "type": t_type, "desc": "Loaded text content"})

        if self.modalities_found["data"]:
            d_type = list[str] if self.is_grouped else str | None
            schema.append({"name": "data_content", "type": d_type, "desc": "Loaded structured data content"})

        if self.modalities_found["image"]:
            i_type = list[pz.ImageFilepath] if self.is_grouped else pz.ImageFilepath | None
            schema.append({"name": "image_filepaths", "type": i_type, "desc": "Paths to image files"})

        if self.modalities_found["audio"]:
            a_type = list[pz.AudioFilepath] if self.is_grouped else pz.AudioFilepath | None
            schema.append({"name": "audio_filepaths", "type": a_type, "desc": "Paths to audio files"})

        super().__init__(id=id, schema=schema)

    def _read_text(self, filepath):
        """ Helper to safely read text content """
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
        
    def _read_pdf(self, filepath):
        """Extracts text using PZ's pdfparser logic."""
        pdf_filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            pdf_bytes = f.read()
        # Use PZ tool to extract text
        return get_text_from_pdf(
            pdf_filename, 
            pdf_bytes, 
            pdfprocessor=self.pdfprocessor, 
            file_cache_dir=self.file_cache_dir
        )
    
    def _read_xls(self, filepath):
        """Reads XLS/XLSX and returns a text representation (CSV) of the first sheet."""
        # We convert to CSV string so it fits into 'data_content' (str)
        df = pd.read_excel(filepath)
        return df.to_csv(index=False)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx: int):
        source_name = self.sources[idx]
        abs_path = os.path.join(self.root_path, source_name)
        
        record = {}
        
        if self.is_grouped:
            record["entity_id"] = source_name
            if self.modalities_found["text"]: record["text_content"] = []
            if self.modalities_found["data"]: record["data_content"] = []
            if self.modalities_found["image"]: record["image_filepaths"] = []
            if self.modalities_found["audio"]: record["audio_filepaths"] = []
            
            files = glob.glob(os.path.join(abs_path, "**"), recursive=True)
            for filepath in sorted(files):
                if os.path.isfile(filepath):
                    _, ext = os.path.splitext(filepath)
                    ext = ext.lower()
                    
                    if ext in self.ext_map["text"]:
                        record["text_content"].append(self._read_text(filepath))
                    elif ext in self.ext_map["data"]:
                        record["data_content"].append(self._read_text(filepath))
                    elif ext in self.ext_map["image"]:
                        record["image_filepaths"].append(filepath)
                    elif ext in self.ext_map["audio"]:
                        record["audio_filepaths"].append(filepath)
        else:
            record["filename"] = source_name
            if self.modalities_found["text"]: record["text_content"] = None
            if self.modalities_found["data"]: record["data_content"] = None
            if self.modalities_found["image"]: record["image_filepaths"] = None
            if self.modalities_found["audio"]: record["audio_filepaths"] = None

            _, ext = os.path.splitext(source_name)
            ext = ext.lower()

            if ext in self.ext_map["text"]:
                if ext in PDF_EXTENSIONS:
                    record["text_content"] = self._read_pdf(abs_path)
                else:
                    record["text_content"] = self._read_text(abs_path)
            elif ext in self.ext_map["data"]:
                if ext in XLS_EXTENSIONS:
                    record["data_content"] = self._read_xls(abs_path)
                else:
                    record["data_content"] = self._read_text(abs_path)
            elif ext in self.ext_map["image"]:
                record["image_filepaths"] = abs_path
            elif ext in self.ext_map["audio"]:
                record["audio_filepaths"] = abs_path

        return record

def load_pz_dataset(dataset_name: str, dataset_path: str) -> pz.Dataset:
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"Registered path for '{ dataset_name }' is not a valid directory: {dataset_path}")

    items_in_root = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path)]
    has_subdirectories = any(os.path.isdir(i) for i in items_in_root)
    if has_subdirectories:
        return MultimodalDataset(id=dataset_name, path=dataset_path)
    
    dataset_extensions = {
        pz.PDFFileDataset: PDF_EXTENSIONS,
        pz.ImageFileDataset: IMAGE_EXTENSIONS,
        pz.AudioFileDataset: AUDIO_EXTENSIONS,
        pz.HTMLFileDataset: HTML_EXTENSIONS,
        pz.XLSFileDataset: XLS_EXTENSIONS,
        pz.TextFileDataset: [".txt", ".md"],
    }
    extension_map = {
        ext.lower(): cls
        for cls, exts in dataset_extensions.items()
        for ext in exts
    }

    found_dataclasses = set()
    files_in_root = [f for f in items_in_root if os.path.isfile(f)]
    
    for f_path in files_in_root:
        _, ext = os.path.splitext(f_path)
        ext = ext.lower()
        if ext in extension_map:
            found_dataclasses.add(extension_map[ext])

    if len(found_dataclasses) == 1:
        DatasetClass = list(found_dataclasses)[0]
        return DatasetClass(id=dataset_name, path=dataset_path)

    elif len(found_dataclasses) > 1:
        return MultimodalDataset(id=dataset_name, path=dataset_path)

    else:
        logger.warning(f"Warning: No compatible file types found for '{dataset_name}'. Defaulting to TextFileDataset.")
        return pz.TextFileDataset(id=dataset_name, path=dataset_path)