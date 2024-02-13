#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to keep global measurements."""
import pandas as pd
import numpy as np

from multipac_testbench.src.instruments.factory import InstrumentFactory
from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint


class GlobalDiagnostics(IMeasurementPoint):
    """Hold measurements unrelated to pick-ups."""

    def __init__(self,
                 name: str,
                 df_data: pd.DataFrame,
                 instrument_factory: InstrumentFactory,
                 instruments_kw: dict,
                 ) -> None:
        """Create the all the global instruments.

        Parameters
        ----------
        df_data : pd.DataFrame
            df_data
        instrument_factory : InstrumentFactory
            An object that creates :class:`.Instrument`.
        instruments_kw : dict[str, dict]
            Dictionary which keys are name of the column where the data from
            the instrument is. Values are dictionaries with keyword arguments
            passed to the proper :class:`.Instrument`.

        """
        super().__init__(name,
                         df_data,
                         instrument_factory,
                         instruments_kw)
        self.position = np.NaN

    def __str__(self) -> str:
        """Give concise info on global diagnostics."""
        out = f"""
        GlobalDiagnostic {self.name},
        with instruments: {[str(x) for x in self.instruments]}
        """
        return " ".join(out.split())
