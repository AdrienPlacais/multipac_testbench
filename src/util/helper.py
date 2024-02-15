#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define general usage functions."""
from pathlib import Path

import numpy as np

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


def r_squared(residue: np.ndarray,
              expected: np.ndarray) -> float:
    """Compute the :math:`R^2` criterion to evaluate a fit.

    For Scipy ``curve_fit`` ``result`` output: ``residue`` is
    ``result[2]['fvec']`` and ``expected`` is the given ``ydata``.

    """
    res_squared = residue**2
    ss_err = np.sum(res_squared)
    ss_tot = np.sum((expected - expected.mean())**2)
    r_squared = 1. - ss_err / ss_tot
    return r_squared
