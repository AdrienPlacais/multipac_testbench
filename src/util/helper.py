#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define general usage functions."""


from pathlib import Path


def output_filepath(filepath: Path,
                    swr: float,
                    out_folder: str,
                    extension: str,
                    ) -> Path:
    """Return a new path to save output files."""
    filename = filepath.with_stem(f"{swr}_"
                                  + filepath.stem).with_suffix(extension)
    output_filepath = filename.parent / out_folder / filename.name
    return output_filepath
