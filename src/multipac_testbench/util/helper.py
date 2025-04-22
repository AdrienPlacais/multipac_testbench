"""Define general usage functions."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def output_filepath(
    filepath: Path,
    swr: float,
    freq_mhz: float,
    out_folder: str | Path,
    extension: str,
) -> Path:
    """Return a new path to save output files.

    Parameters
    ----------
    filepath :
        Name of the data ``CSV`` file from LabViewer.
    swr :
        Theoretical :math:`SWR` to add to the output file name.
    freq_mhz :
        Theoretical rf frequency to add to the output file name.
    out_folder :
        Relative name of the folder where data will be saved; it is defined
        w.r.t. to the parent folder of ``filepath``.
    extension :
        Extension of the output file, with the dot.

    Returns
    -------
    filename :
        A full filepath.

    """
    if np.isinf(swr):
        swr_str = "SWR_infty"
    else:
        swr_str = f"SWR_{int(swr):05.0f}"
    freq_str = f"freq_{freq_mhz:03.0f}MHz"

    filename = (
        filepath.with_stem(("_").join((swr_str, freq_str, filepath.stem)))
        .with_suffix(extension)
        .name
    )

    folder = filepath.parent / out_folder

    if not folder.is_dir():
        folder.mkdir(parents=True)

    return folder / filename


def r_squared(
    residue: NDArray[np.float64], expected: NDArray[np.float64]
) -> float:
    """Compute the :math:`R^2` criterion to evaluate a fit.

    For Scipy ``curve_fit`` ``result`` output: ``residue`` is
    ``result[2]['fvec']`` and ``expected`` is the given ``data``.

    """
    res_squared = residue**2
    ss_err = np.sum(res_squared)
    ss_tot = np.sum((expected - expected.mean()) ** 2)
    r_squared = 1.0 - ss_err / ss_tot
    return r_squared
