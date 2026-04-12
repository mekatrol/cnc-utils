from dataclasses import dataclass


@dataclass
class FileLocations:
    load_project_at_startup: bool = True
    recent_project_count: int = 10
    last_load_project_directory: str = ""
    last_load_directory: str = ""
    last_save_directory: str = ""
    recent_projects: list[str] | None = None

    def __post_init__(self) -> None:
        if self.recent_projects is None:
            self.recent_projects = []
