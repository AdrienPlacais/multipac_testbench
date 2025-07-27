"""Define helper functions to resolve paths and load files."""

import logging
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the configuration file."""
    with open(resolve_path(config_path), "rb") as f:
        config = tomllib.load(f)
    return config


def resolve_path(path: str | Path) -> Path:
    """Try to resole the given path.

    1. If it is absolute, we take it as is.
    2. If not, we look if it is in working dir.
    3. If not, we look if it is in packaged data.

    """
    if isinstance(path, str):
        path = Path(path)

    if path.is_absolute():
        if not path.exists():
            logging.error(
                f"You gave {path = }, which is absolute and does not exists. I"
                " will return it anyway, but errors are expected."
            )
        return path

    if path.exists():
        return path

    package_data_root = resources.files("multipac_testbench.data")
    fallback = package_data_root / str(path)
    if fallback.is_file():
        return fallback

    logging.error(
        f"You gave {path = }, which was not found to be an absolute file, a "
        f"relative existing file. It is neither packaged data ({fallback} not "
        "found). Continuing anyway, errors are expected."
    )
    return path
