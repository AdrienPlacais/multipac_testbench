"""Define helper funcs for :class:`.MultipactorTest`, :class:`.PowerStep`."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def parse_header_value(commented_lines: Sequence[str], key: str) -> float:
    """Extract a scalar value from the ``CSV`` commented header.

    Parameters
    ----------
    commented_lines :
        Lines from the file header, already stripped of their comment
        character.
    key :
        The header key to look up (e.g. ``"Polarisation_2"``).

    Returns
    -------
        The parsed float value, or ``np.nan`` if the key is absent or the
        value cannot be converted (including ``"NaN"`` entries).

    """
    for line in commented_lines:
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[0].strip() == key:
            try:
                return float(parts[1].strip())
            except ValueError:
                return np.nan
    logging.warning(f"{key = } not found in header. Returning NaN.")
    return np.nan


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
