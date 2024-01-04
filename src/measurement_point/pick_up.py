#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to keep measurements at a pick-up."""

import pandas as pd

from multipac_testbench.src.instruments.factory import InstrumentFactory
from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint


class PickUp(IMeasurementPoint):
    """Hold measurements at a single pick-up."""

    def __init__(self,
                 name: str,
                 df_data: pd.DataFrame,
                 instrument_factory: InstrumentFactory,
                 position: float,
                 instruments_kw: dict,
                 ) -> None:
        """Create the pick-up with all its instruments.

        Parameters
        ----------
        name : str
            Name of the pick-up.
        df_data : pd.DataFrame
            df_data
        instrument_factory : InstrumentFactory
            An object that creates :class:`.Instrument`.
        position : float
            position
        instruments_kw : dict[str, dict]
            Dictionary which keys are name of the column where the data from
            the instrument is. Values are dictionaries with keyword arguments
            passed to the proper :class:`.Instrument`.

        """
        super().__init__(df_data, instrument_factory, instruments_kw)
        self.name = name
        self.position = position

    def __str__(self) -> str:
        """Give concise info on pick-up."""
        out = f"""
        Pick-Up {self.name} at z = {self.position:1.3f}m,
        with instruments: {[str(x) for x in self.instruments]}
        """
        return " ".join(out.split())
