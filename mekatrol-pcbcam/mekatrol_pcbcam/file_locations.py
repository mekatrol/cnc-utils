from dataclasses import dataclass


@dataclass
class FileLocations:
    last_load_directory: str = ""
    last_save_directory: str = ""
