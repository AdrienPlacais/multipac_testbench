"""Define helper funcs for :class:`.MultipactorTest`, :class:`.PowerStep`."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def infer_dbm(commented_lines: Sequence[str]) -> float:
    """Determine :unit:`dBm` of current step from power step file header."""
    if len(commented_lines) < 7:
        raise ValueError(
            "There is too few lines in the headers. We expect the dBm to be "
            "on the 7th line."
        )
    expected_line = commented_lines[6]
    splitted = expected_line.split()
    if splitted[0] != "SM300_Level":
        raise ValueError(
            "We expect the 7th header line to start with 'SM300_Level'."
        )
    return float(splitted[-1])


#: Functions converting an :meth:`.Instrument._raw_data` to a single float
#: value
REDUCER_T = Callable[[NDArray], float]


def take_maximum(raw_data: NDArray) -> float:
    """Take the maximum of the array.

    This is the default behavior for LabViewer.

    """
    value = np.max(raw_data)
    if np.isnan(value):
        logging.warning("NaN detected. Returning highest float instead.")
        value = np.nanmax(raw_data)
    return float(value)


def take_median(
    raw_data: NDArray, first_index: int = 0, last_index: int = -1
) -> float:
    """Take median from ``first_index`` to ``last_index``."""
    size = len(raw_data)
    try:
        sample = raw_data[first_index:last_index]
    except IndexError:
        logging.error(
            f"raw_data has length {size}, so accessing the slice {first_index}"
            f":{last_index} raised an error. Taking everything instead."
        )
        sample = raw_data

    value = np.median(sample)
    return float(value)


#: Functions detecting if the file as argument corresponds to a
#: :class:`.PowerStep` file.
POWERSTEP_FILE_RECOGNIZER_T = Callable[[Path], bool]


def default_powerstep_file_valider(path: Path) -> bool:
    """Detect ``CSV`` files."""
    if path.suffix != ".csv":
        return False
    return True


def powerstep_files(
    folder: Path, file_recognizer: POWERSTEP_FILE_RECOGNIZER_T
) -> dict[Path, int]:
    """Gather powerstep files in ```folder`` with their ``sample_index``.

    Parameters
    ----------
    folder :
        Directory holding all the power step files of a test.
    file_recognizer :
        Takes in a path, determine if it should be loaded.

    Returns
    -------
        Maps power step files with corresponding sample index.

    """
    files = sorted(path for path in folder.iterdir() if file_recognizer(path))
    file_index_mapping = {f.absolute(): i for i, f in enumerate(files)}
    return file_index_mapping
