"""Define functions to prepare data for :class:`.MultipactorTest`."""

import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: How consecutive-same power points should be treated.
#:
#: - ``"keep_all"``: keep all data (default)
#: - ``"trim"``: remove trailing points
#: - ``"average"``: average the data on the same point
#: - ``"first"``: only consider first point (least conditionned)
#: - ``"last"``: only consider last point (most conditionned)
#: - ``"max"``: retain maximum value
#: - ``"min"``: retain minimum value
#:
TRIGGER_POLICIES = Literal[
    "keep_all", "trim", "average", "first", "last", "max", "min"
]


def load(
    filepath: Path,
    sep: str = "\t",
    trigger_policy: TRIGGER_POLICIES = "keep_all",
    index_col: str = "Sample index",
    remove_metadata_columns: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, list[str]]:
    """Load the LabViewer file.

    If ``trigger_policy`` is set, perform operations to select the desired
    trigger. These operations do not preserve original sample indexes.

    Parameters
    ----------
    filepath :
        LabViewer file to be loaded.
    sep :
        Column separator.
    trigger_policy :
        How consecutive measures at the same power should be treated.
    index_col :
        Name of the column holding indexes.
    remove_metadata_columns :
        Remove the rightmost columns holding metadata.
    kwargs :
        Other kwargs passed to :func:`._load_file`.

    Returns
    -------
    pandas.DataFrame
        Holds data.
    list[str]
        The comments, without their comment character, line by line. If loading
        a ``XLSX``, an empty list is returned.

    """
    data, commented_lines = _load_file(
        filepath.resolve(), sep=sep, index_col=index_col, **kwargs
    )

    if remove_metadata_columns:
        data = data.select_dtypes(include=["float", "int"])

    filtered = _apply_trigger_filtering(
        trigger_policy, data, dbm_column="NI9205_dBm"
    ).reset_index(drop=True)
    filtered.index.name = index_col

    printer = logging.info
    if trigger_policy in ("average", "keep_all"):
        printer(f"Applied {trigger_policy = } on {filepath}")
        return filtered, commented_lines

    fraction = 100 * len(filtered) / len(data)
    if trigger_policy == "trim":
        if fraction < 90.0 and trigger_policy:
            printer = logging.warning
        elif fraction < 50.0:
            printer = logging.error

    printer(f"After {trigger_policy = }, kept {fraction:.2f}% of {filepath}")
    return filtered, commented_lines


def _load_file(
    filepath: Path,
    index_col: str = "Sample index",
    comment: str = "#",
    **kwargs,
) -> tuple[pd.DataFrame, list[str]]:
    """Load the data file.

    .. todo::
        Allow for ``TXT`` or ``XLSX`` input files.

    Parameters
    ----------
    filepath :
        File to load.
    index_col :
        Name of the index column.
    comment :
        Comment character.
    kwargs :
        Other keyword arguments passed to the loading function. Holds
        ``"sep"`` key-value pair, which is removed if loading a ``XLSX``.

    Returns
    -------
    pandas.DataFrame
        Holds data.
    list[str]
        The comments, without their comment character, line by line. If loading
        a ``XLSX``, an empty list is returned.

    """
    ext = filepath.suffix
    if ext == ".csv":
        pandas_reader = pd.read_csv
    elif ext == ".xlsx":
        pandas_reader = pd.read_excel
        if "sep" in kwargs:
            del kwargs["sep"]
    else:
        logging.error(f"{filepath} extension not supported.")
        raise RuntimeError
    try:
        data = pandas_reader(
            (filepath), index_col=index_col, comment=comment, **kwargs
        )
    except Exception as e:
        logging.error(
            f"There was a mismatch in the number of columns in {filepath}"
            ". Check that the number of column header match the number of "
            "columns, that the trailing comments in the first lines "
            "were removed, and that you set the appropriate column separator."
        )
        logging.exception(e)
        raise e

    if ext == ".xlsx":
        return data, []

    commented_lines = []
    with open(filepath) as f:
        for line in f:
            if line.startswith(comment):
                commented_lines.append(line.removeprefix(comment).rstrip("\n"))
            else:
                break

    return data, commented_lines


def _apply_trigger_filtering(
    trigger_policy: TRIGGER_POLICIES,
    data: pd.DataFrame,
    dbm_column: str = "NI9205_dBm",
    tol: float = 1e-10,
) -> pd.DataFrame:
    """Apply desired trigger policy.

    Original indexes are not preserved.

    """
    if trigger_policy == "keep_all":
        return data

    if dbm_column not in data.columns:
        logging.error(
            f"{dbm_column = } not found in the results file. Mandatory for "
            "edition of trigger."
        )
        return data

    power = data[dbm_column].to_numpy()
    labels = _group_consecutive_equal_power(power, tol)
    grouped = data.groupby(labels, sort=False)

    if trigger_policy == "trim":
        unique_labels = np.unique(labels)
        mask = (labels != unique_labels[0]) & (labels != unique_labels[-1])
        trimmed = data[mask]
        return trimmed

    if trigger_policy == "average":
        return grouped.mean(numeric_only=True)

    if trigger_policy == "first":
        return grouped.nth(0)

    if trigger_policy == "last":
        return grouped.nth(-1)

    logging.error(f"{trigger_policy = } not understood. Not doing anything.")
    return data


def _group_consecutive_equal_power(
    power: NDArray, tol: float = 1e-10
) -> NDArray[np.int32]:
    """Gather measurements with the same power (consecutive).

    Parameters
    ----------
    power :
        The input power array.
    tol :
        Tolerance for comparing equality.

    Returns
    -------
        An array of group labels of the same length as ``power``.

    """
    labels = [0]
    group = 0
    for i in range(1, len(power)):
        if not math.isclose(power[i], power[i - 1], abs_tol=tol):
            group += 1
        labels.append(group)
    return np.array(labels)


def save(
    filepath: Path,
    data: pd.DataFrame,
    info: str | None = None,
    verbose=True,
    **kwargs,
) -> None:
    """Save the dataframe as a new LabViewer results file."""
    save_meth = data.to_csv
    if filepath.suffix == ".xlsx":
        save_meth = data.to_excel
        if "sep" in kwargs:
            del kwargs["sep"]

    if info is None:
        info = "new file"

    if verbose:
        logging.info(f"Saving {info} to {filepath}")
    try:
        save_meth(filepath, **kwargs)
    except OSError as e:
        logging.error(f"Could not save file:\n{e}")
