"""Define an object to keep global measurements."""

import numpy as np
import pandas as pd
from multipac_testbench.instruments.factory import InstrumentFactory
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)


class GlobalDiagnostics(IMeasurementPoint):
    """Hold measurements unrelated to pick-ups."""

    def __init__(
        self,
        name: str,
        df_data: pd.DataFrame,
        instrument_factory: InstrumentFactory,
        instruments_kw: dict,
    ) -> None:
        """Create the all the global instruments.

        Parameters
        ----------
        df_data :
            df_data
        instrument_factory :
            An object that creates :class:`.Instrument`.
        instruments_kw :
            Dictionary which keys are name of the column where the data from
            the instrument is. Values are dictionaries with keyword arguments
            passed to the proper :class:`.Instrument`.

        """
        super().__init__(
            name, df_data, instrument_factory, instruments_kw, position=np.nan
        )

    def __str__(self) -> str:
        """Give concise info on global diagnostics."""
        out = f"""
        GlobalDiagnostic {self.name},
        with instruments: {[str(x) for x in self.instruments]}
        """
        return " ".join(out.split())
