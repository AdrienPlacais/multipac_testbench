"""Define an object corresponding to a power step file."""

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from multipac_testbench.multipactor_test import MultipactorTest
from numpy.typing import NDArray

REDUCER_T = Callable[[NDArray], float]


def _infer_dbm(filepath: Path) -> float:
    """Determine the dBm of current step from filename."""
    filename = filepath.name
    left_delim = "("
    right_delim = " dBm)"
    for delim in (left_delim, right_delim):
        assert (
            delim in filename
        ), f"Need a {delim} character in {filename = } to determine dBm."

    try:
        dbm = filename.split(left_delim)[1].split(right_delim)[0]
    except Exception as e:
        logging.critical(
            f"An exception was raised trying to split {filename = }. Returning"
            f" 0dBm and hoping for the best. Exception:\n{e}"
        )
        return 0.0

    try:
        value = float(dbm)
    except Exception as e:
        logging.critical(
            f"An exception was raised trying to convert {dbm = } to float. "
            f"Returning 0dBm and hoping for the best. Exception:\n{e}"
        )
        return 0.0
    return value


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
    raw_data: NDArray, first_index: int = -100, last_index: int = -1
) -> float:
    """Take meian from ``first_index`` to ``last_index``."""
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


class PowerStep(MultipactorTest):
    """This object is basically a MultipactorTest. But for one power step."""

    def __init__(
        self,
        filepath: Path,
        config: dict,
        freq_mhz: float,
        swr: float,
        sample_index: int,
        sep: str = "\t",
        index_col: str = "Index",
        dbm: float | None = None,
        out_dbm_column: str = "NI9205_dBm",
        out_index_col: str = "Sample index",
        **kwargs,
    ) -> None:
        """Create object like if it was a :class:`.MultipactorTest`.

        The differences are:

        - ``index_col`` is by default ``"Index"``, like in the ``MV`` files.
        - ``trigger_policy`` is always ``"keep_all"``, as other values would be
          meaningless.

        Parameters
        ----------
        filepath :
            Path to the results file produced by LabViewer.
        config :
            Configuration ``TOML`` of the testbench.
        freq_mhz :
            Frequency of the test in :unit:`MHz`.
        swr :
            Expected Voltage Signal Wave Ratio.
        sample_index :
            Index of power step.
        sep :
            Delimiter between two columns in ``filepath``.
        index_col :
            Name of the column holding index data.
        out_index_col :
            Where to store ``sample_index`` in the output file.
        dbm :
            To override the dBm values inferred from filename.
        out_dbm_column :
            Where to store the dBm value in the output file.
        kwargs :
            Other kwargs passed to :func:`.load`.

        """
        super().__init__(
            filepath=filepath,
            config=config,
            freq_mhz=freq_mhz,
            swr=swr,
            info="",
            sep=sep,
            index_col=index_col,
            trigger_policy="keep_all",
            **kwargs,
        )
        self._sample_index = sample_index
        self._out_index_col = out_index_col
        self._dbm = _infer_dbm(filepath) if dbm is None else dbm
        self._out_dbm_column = out_dbm_column

    def to_single_values(self, reducer: REDUCER_T) -> pd.Series:
        """Convert arrays of :class:`.Instrument` values to single floats.

        Parameters
        ----------
        reducer :
            Function converting array to float. The default in LabViewer is to
            take the maximum.

        """
        series = self.df_data.apply(reducer, axis=0, raw=True)
        series[self._out_dbm_column] = self._dbm
        series[self._out_index_col] = self._sample_index
        return series


def create_multipactor_test_file(
    power_steps: Iterable[PowerStep],
    csv_path: Path,
    reducer: REDUCER_T,
    index_col: str = "Sample index",
    **kwargs,
) -> None:
    """Create a file that can be loaded by :class:`MultipactorTest`.

    Parameters
    ----------
    power_steps :
        All the power steps of the file.
    csv_path :
        Where the resulting ``CSV`` will be stored.
    reducer :
        Function converting array to float. The default in LabViewer is to take
        the maximum.
    index_col :
        Name of the column that will contain each power step index.

    """
    series = (
        power_step.to_single_values(reducer)
        for power_step in sorted(
            power_steps, key=lambda step: step._sample_index
        )
    )
    df = pd.concat(series, axis=1).transpose().set_index(index_col)
    df.to_csv(csv_path, **kwargs)
    logging.info(f"MultipactorTest file saved to {csv_path}")
    return
