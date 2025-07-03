#!/usr/bin/env python3
"""Define an object corresponding to a power step file."""

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd
from multipac_testbench.multipactor_test import MultipactorTest
from multipac_testbench.multipactor_test.helper import (
    POWERSTEP_FILE_RECOGNIZER_T,
    REDUCER_T,
    default_powerstep_file_valider,
    infer_dbm,
    powerstep_files,
    take_maximum,
)


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

        Keys such as ``freq_mhz`` or ``swr`` are not used to create
        :class:`.MultipactorTest` files; however the let you perform
        :meth:`PowerStep.sweet_plot`.

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
            remove_metadata_columns=True,
            **kwargs,
        )
        self._sample_index = sample_index
        self._out_index_col = out_index_col
        self._dbm = infer_dbm(filepath) if dbm is None else dbm
        self._out_dbm_column = out_dbm_column

    def to_single_values(self, reducer: REDUCER_T) -> pd.Series:
        """Convert arrays of :class:`.Instrument` values to single floats.

        Parameters
        ----------
        reducer :
            Function converting array to float. The default in LabViewer is to
            take the maximum.

        """
        __import__("pdb").set_trace()
        series = self.df_data.apply(reducer, axis=0, raw=True)
        series[self._out_dbm_column] = self._dbm
        series[self._out_index_col] = self._sample_index
        return series


class PowerStepSet:
    """Define all the files consituting a :class:`.MultipactorTest`."""

    def __init__(
        self,
        folder: Path,
        config: dict,
        freq_mhz: float,
        swr: float,
        sep: str = "\t",
        index_col: str = "Index",
        dbms: Mapping[str, float] | None = None,
        out_dbm_column: str = "NI9205_dBm",
        out_index_col: str = "Sample index",
        file_recognizer: POWERSTEP_FILE_RECOGNIZER_T | None = None,
        **kwargs,
    ) -> None:
        """Load all ``MV`` files in ``folder``, create :class:`.PowerStep`.

        Parameters
        ----------
        folder :
            Directory holding all the power step files of a test.
        config :
            Configuration file for the test.
        freq_mhz :
            RF frequency in :unit:`.MHz`.
        swr :
            SWR of the test.
        sep :
            Column delimiter.
        index_col :
            Name of the column holding indexes in every power step file.
        dbms :
            Maps filenames to their :unit:`dBm`, when they are shifted.
        out_dbm_column :
            Name of column where power setpoint in :unit:`dBm` will be stored.
        out_index_col :
            Name of column where sample indexes will be stored.
        file_recognizer :
            Function taking in a filepath, and determining if it is a valid
            power step file. If not provided, set to
            :func:`.default_powerstep_file_valider`.

        """
        self._folder = folder
        file_recognizer = (
            file_recognizer
            if file_recognizer
            else default_powerstep_file_valider
        )
        file_index_mapping = powerstep_files(folder, file_recognizer)

        self._power_steps = [
            PowerStep(
                filepath=filepath,
                config=config,
                freq_mhz=freq_mhz,
                swr=swr,
                sample_index=sample_index,
                sep=sep,
                index_col=index_col,
                dbm=(
                    dbms.get(filepath.name, None) if dbms is not None else None
                ),
                out_dbm_column=out_dbm_column,
                out_index_col=out_index_col,
                **kwargs,
            )
            for filepath, sample_index in file_index_mapping.items()
        ]
        if len(self) == 0:
            logging.warning(f"No valid file found in {folder}")

    def __iter__(self) -> Iterator[PowerStep]:
        """Iterate over :class:`.PowerStep` objects.

        Yields
        ------
        PowerStep
            The stored :class:`.PowerSample` objects, sorted by sample index.

        """
        return iter(self._power_steps)

    def __str__(self) -> str:
        """Print out origin folder, number of loaded files."""
        return f"PowerStepSet holding {len(self)} files from {self._folder}"

    def __len__(self) -> int:
        """Get number of loaded files."""
        return len(self._power_steps)

    def to_multipactor_test_file(
        self,
        csv_path: Path,
        reducer: REDUCER_T | None = None,
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
            Function converting array to float. The default in LabViewer is to
            take the maximum. If not set, we also use this.
        index_col :
            Name of the column that will contain each power step index.

        """
        __import__("pdb").set_trace()
        series = (
            power_step.to_single_values(
                reducer if reducer is not None else take_maximum
            )
            for power_step in sorted(self, key=lambda step: step._sample_index)
        )
        df = pd.concat(series, axis=1).transpose().set_index(index_col)
        df.to_csv(csv_path, **kwargs)
        logging.info(f"MultipactorTest file saved to {csv_path}")
        return


if __name__ == "__main__":
    import tomllib

    dir = (Path(__file__).parents[1] / "data/power_step_files/").resolve()
    config_path = Path(dir, "testbench_configuration.toml")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    power_step_set = PowerStepSet(dir, config, freq_mhz=160.0, swr=4.0)
    power_step_set.to_multipactor_test_file(dir / "test.csv")
