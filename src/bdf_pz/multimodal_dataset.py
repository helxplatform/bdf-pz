import palimpzest as pz
import glob
import os

class MultimodalDataset(pz.IterDataset):
    def __init__(self, id: str, path: str):
        self.root_path = path
        
        root_items = [os.path.join(path, i) for i in os.listdir(path)]
        self.is_grouped = any(os.path.isdir(i) for i in root_items)

        # Map modalities to supported extensions
        self.ext_map = {
            "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"],
            "audio": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
            "text":  [".txt", ".md", ".html", ".htm", ".xml"],
            "data":  [".csv", ".json", ".yaml", ".xls", ".xlsx"]
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
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

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
                record["text_content"] = self._read_text(abs_path)
            elif ext in self.ext_map["data"]:
                record["data_content"] = self._read_text(abs_path)
            elif ext in self.ext_map["image"]:
                record["image_filepaths"] = abs_path
            elif ext in self.ext_map["audio"]:
                record["audio_filepaths"] = abs_path

        return record