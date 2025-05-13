"""Define general usage functions."""

from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

T = TypeVar("T")


def is_nested_list(obj: list[T] | list[list[T]]) -> bool:
    """Tell if ``obj`` is a nested list."""
    return bool(obj) and isinstance(obj[0], list)


def split_rows_by_mask(
    df: pd.Series | pd.DataFrame,
    mask: NDArray[np.bool],
    suffixes: tuple[str, str] = ("__increasing", "__decreasing"),
) -> pd.DataFrame:
    """
    Split the rows of a Series or DataFrame into two new columns based on a boolean mask.

    For each original column, two new columns are created:
    one containing the values where ``mask`` is True, and another where it is False.
    The new columns are named by appending the given suffixes.

    Examples
    --------
    >>> mask = np.array([True, False, True])
    >>> suffixes = ("__a", "__b")
    >>> ser = pd.Series([1, 2, 3])
    >>> print(split_rows_by_mask(ser, mask, suffixes))
       __a  __b
    0  1.0  NaN
    1  NaN  2.0
    2  3.0  NaN

    >>> df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    >>> print(split_rows_by_mask(df, mask, suffixes))
       col1__a  col1__b  col2__a  col2__b
    0      1.0      NaN      4.0      NaN
    1      NaN      2.0      NaN      5.0
    2      3.0      NaN      6.0      NaN

    Parameters
    ----------
    df :
        The input data to split row-wise.
    mask :
        A boolean array of the same length as ``df``. True values go to the
        first output column, False to the second.
    suffixes :
        Suffixes to append to column names for True and False rows,
        respectively.

    Returns
    -------
        A new DataFrame with columns split according to the mask.

    """
    if len(df) != len(mask):
        raise ValueError("Length of mask must match number of rows in df")

    df = df.to_frame() if isinstance(df, pd.Series) else df
    mask_df = pd.DataFrame(
        np.broadcast_to(mask[:, None], df.shape),
        index=df.index,
        columns=df.columns,
    )

    df_true = df.where(mask_df)
    df_false = df.where(~mask_df)

    new_cols = {}
    for col in df.columns:
        new_cols[f"{col}{suffixes[0]}"] = df_true[col]
        new_cols[f"{col}{suffixes[1]}"] = df_false[col]

    return pd.DataFrame(new_cols, index=df.index)


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
