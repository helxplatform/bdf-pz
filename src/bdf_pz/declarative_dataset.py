import os
import yaml
import palimpzest as pz
import re
from typing import TypedDict, Literal, Any
from glob import glob

class SchemaField(TypedDict):
    name: str
    type: Any
    desc: str

class GroupKeyField(TypedDict):
    """ A field whose value is the grouping key itself (e.g., a subdirectory name). """
    name: str
    source: Literal["group_key"]

class FileBasedField(TypedDict):
    """ Base definition for a field derived from one or more files. """
    name: str
    source: Literal["content", "filepath"]
    glob: str | list[str] # Glob pattern(s) to find the relevant file(s).
    cardinality: Literal["one", "many"]

LoadingField = GroupKeyField | FileBasedField

class SubdirectoryLoadingPlan(TypedDict):
    """ Loading strategy where each subdirectory constitutes one data record. """
    strategy: Literal["by_subdirectory"]
    fields: list[LoadingField]

class FilenamePatternLoadingPlan(TypedDict):
    """ Loading strategy where files are grouped by a shared filename pattern. """
    strategy: Literal["by_filename_pattern"]
    pattern: str  # Regex with one capture group to extract the {group_key}.
    fields: list[LoadingField]

LoadingPlan = SubdirectoryLoadingPlan | FilenamePatternLoadingPlan

class DatasetSpec(TypedDict):
    """ A standardized spec supporting the loading of an unstructured, multimodal dataset into Palimpzest. """
    schema: list[SchemaField]
    loading_plan: LoadingPlan


class DeclarativeDataset(pz.IterDataset):
    """
    A generic dataset loader that configures itself based on a dataset spec
    """
    def __init__(self, id: str, path: str, spec: dict, **kwargs):
        self.root_dir = path
        self.spec = spec

        super().__init__(id=id, schema=self.spec["schema"])

        # Discover all data records based on the specified strategy
        self._discover_records()

    def _discover_records(self):
        self.records = []
        strategy = self.spec["loading_plan"]["strategy"]

        if strategy == 'by_subdirectory':
            self.records = sorted(
                [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
            )
        elif strategy == 'by_filename_pattern':
            pattern = re.compile(self.spec['loading_plan']['pattern'])
            group_keys = set()
            for filename in os.listdir(self.root_dir):
                match = pattern.match(filename)
                if match:
                    group_keys.add(match.group(1)) # Add the captured group
            self.records = sorted(list(group_keys))
        else:
            raise ValueError(f"Unknown loading strategy: {strategy}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        group_key = self.records[idx]
        record = {}

        # Determine the search path (root dir or a subdirectory)
        search_path = self.root_dir
        if self.spec['loading_plan']['strategy'] == 'by_subdirectory':
            search_path = os.path.join(self.root_dir, group_key)

        # Populate fields based on the loading plan
        for field in self.spec['loading_plan']['fields']:
            field_name = field['name']
            source = field['source']

            if source == 'group_key':
                record[field_name] = group_key
                continue
            
            # Find files using the glob pattern
            glob_pattern = field['glob']
            if isinstance(glob_pattern, str):
                glob_pattern = [glob_pattern]

            found_files = []
            for pattern in glob_pattern:
                # Substitute {group_key} if present
                formatted_pattern = pattern.format(group_key=group_key)
                found_files.extend(glob(os.path.join(search_path, formatted_pattern)))

            # Handle cardinality
            if field['cardinality'] == 'one':
                if len(found_files) != 1:
                    # Handle error: wrong number of files found (e.g., log a warning)
                    value = None
                else:
                    filepath = found_files[0]
                    if source == 'filepath':
                        value = filepath
                    elif source == 'content':
                        with open(filepath, 'r', encoding='utf-8') as f:
                            value = f.read()
                    else:
                        value = None # Should not happen with valid spec
                record[field_name] = value

            elif field['cardinality'] == 'many':
                values = []
                for filepath in sorted(found_files):
                    if source == 'filepath':
                        values.append(filepath)
                    elif source == 'content':
                        with open(filepath, 'r', encoding='utf-8') as f:
                            values.append(f.read())
                record[field_name] = values
        
        return record