"""Define general usage functions."""

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

T = TypeVar("T")


def is_nested_list(obj: list[T] | list[list[T]]) -> bool:
    """Tell if ``obj`` is a nested list."""
    return bool(obj) and isinstance(obj[0], list)


def flatten[T](nest: Iterable[T]) -> Iterator[T]:
    """Flatten nested list of lists of..."""
    for _in in nest:
        if isinstance(_in, Iterable) and not isinstance(_in, (str, bytes)):
            yield from flatten(_in)
        else:
            yield _in


def split_rows_by_masks(
    df: pd.Series | pd.DataFrame,
    masks: dict[str, NDArray[np.bool]],
) -> pd.DataFrame:
    """Split the rows of ``df`` into new columns based on a boolean mask.

    For each column in the original data, one new column per mask is created
    with the corresponding suffix. Rows not selected by a mask are filled with
    ``np.nan``.

    .. important::
        Functions using the splitted ``df`` such as
        :func:`.styles_from_column_cycle` expect every key of ``masks`` to
        start with a double underscore (``__``).

    Examples
    --------
    >>> mask = np.array([True, False, True])
    >>> masks = {"__(grows)": mask, "__(decreases)": ~mask}
    >>> ser = pd.Series([1, 2, 3], name=data)
    >>> print(split_rows_by_masks(ser, masks))
       data__(grows) data__(decreases)
    0  1.0           NaN
    1  NaN           2.0
    2  3.0           NaN

    >>> df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    >>> print(split_rows_by_masks(df, masks))
       col1__(grows)  col1__(decreases)  col2__(grows)  col2__(decreases)
    0  1.0            NaN                4.0            NaN
    1  NaN            2.0                NaN            5.0
    2  3.0            NaN                6.0            NaN


    Raises
    ------
    ValueError
        If any row is matched by more than one mask or if mask lengths do not
        match the input.

    Parameters
    ----------
    df :
        The input data to split row-wise.
    masks :
        A dictionary where each key is a suffix used to label the split
        columns, and each value is a boolean mask of the same length as the
        input data. Keys must start with two underscores `(`__``) to enable
        consistent column naming and compatibility with downstream styling
        logic (e.g., grouping lines by base column in plots). If multiple masks
        are ``True`` at the same row index, a ``ValueError`` is raised.

    Returns
    -------
        A new DataFrame with columns split according to the masks.

    """
    if not masks:
        raise ValueError("At least one mask must be provided.")

    length = len(df)
    for name, mask in masks.items():
        if len(mask) != length:
            raise ValueError(
                f"Mask '{name}' has incorrect length ({len(mask)} != {length})"
            )

    for key in masks.keys():
        if key[:2] == "__":
            continue
        logging.warning(
            f"{key = } does not start with a double underscore. Splitted "
            "columns may not be recognized."
        )

    # Ensure disjoint masks
    combined = np.zeros(length, dtype=int)
    for mask in masks.values():
        combined += mask.astype(int)
    if (combined > 1).any():
        raise ValueError(
            "Masks must be disjoint: multiple masks are True at the same "
            "position."
        )
    df = df.to_frame() if isinstance(df, pd.Series) else df
    result = {}

    for col in df.columns:
        for suffix, mask in masks.items():
            col_name = f"{col}{suffix}"
            result[col_name] = df[col].where(mask)

    return pd.DataFrame(result)


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


def types(my_list: Sequence) -> set[type]:
    """Get all different types in given list."""
    return {type(x) for x in my_list}


def types_match(my_list: Sequence, to_match: type) -> bool:
    """Check if all elements of ``my_list`` have type ``type``."""
    return types(my_list) == {to_match}


def drop_repeated_col(
    df: pd.DataFrame, col: pd.Index | str | None = None
) -> pd.DataFrame:
    """Remove consecutive rows with the same ``col`` value.

    If ``x_column`` is not provided, we take the first column in the dataframe.

    This function is used with :class:`.RPACurrent` and :class:`.RPAPotential`
    data.

    """
    if col is None:
        col = df.columns[0]
    df = df.loc[df[col] != df[col].shift()]
    return df
