#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define general usage functions."""


from pathlib import Path


def output_filepath(filepath: Path,
                    swr: float,
                    freq_mhz: float,
                    out_folder: str,
                    extension: str,
                    ) -> Path:
    """Return a new path to save output files."""
    filename = filepath.with_stem(f"swr{swr}_freq{freq_mhz}"
                                  + filepath.stem).with_suffix(extension).name
    folder = filepath.parent / out_folder
    if not folder.is_dir():
        folder.mkdir(parents=True)
    return folder / filename
