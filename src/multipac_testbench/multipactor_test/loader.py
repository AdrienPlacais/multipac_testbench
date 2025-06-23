"""Define functions to prepare data for :class:`.MultipactorTest.`"""

import logging
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
    dbm_column: str = "NI9205_dBm",
    **kwargs,
) -> pd.DataFrame:
    """Load the LabVIEWER file.

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
    dbm_column :
        Name of the column holding the power.
    kwargs :
        Other kwargs passed to :func:`pandas.load`.

    """
    data = _load_file(filepath, sep=sep, **kwargs)

    filtered = _apply_trigger_filtering(
        trigger_policy, data, dbm_column=dbm_column
    ).reset_index()

    printer = logging.info
    if trigger_policy in ("average", "keep_all"):
        printer(f"Applied {trigger_policy = } on {filepath}")
        return filtered

    fraction = 100 * len(filtered) / len(data)
    if trigger_policy == "trim":
        if fraction < 90.0 and trigger_policy:
            printer = logging.warning
        elif fraction < 50.0:
            printer = logging.error

    printer(f"After {trigger_policy = }, kept {fraction:.2f}% of {filepath}")
    return filtered


def _load_file(filepath: Path, **kwargs) -> pd.DataFrame:
    """Load the data file.

    .. todo::
        Allow for ``TXT`` or ``XLSX`` input files.

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
        data = pandas_reader(filepath, index_col="Sample index", **kwargs)
    except Exception as e:
        logging.error(
            f"There was a mismatch is the number of columns in {filepath}"
            ". Check that the number of column header match the number of "
            "columns, and that the trailing comments in the first lines "
            "were removed."
        )
        logging.exception(e)
        raise e

    return data


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
    NDArray
        An array of group labels of the same length as ``power``.

    """
    labels = [0]
    group = 0
    for i in range(1, len(power)):
        if abs(power[i] - power[i - 1]) >= tol:
            group += 1
        labels.append(group)
    return np.array(labels)
