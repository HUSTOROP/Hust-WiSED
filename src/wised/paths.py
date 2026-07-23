from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_project_root_on_path() -> None:
    """Make project-root modules importable for the unified runner."""
    for path in (PROJECT_ROOT, SRC_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def task_output_dir(base_dir: Union[str, Path], task_name: str) -> Path:
    """Return the standard per-task output directory."""
    return Path(base_dir) / str(task_name)
