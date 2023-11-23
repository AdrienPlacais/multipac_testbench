#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to create the proper :class:`.Instrument`."""
from typing import Any

import pandas as pd

from multipac_testbench.src.instruments.current_probe import CurrentProbe
from multipac_testbench.src.instruments.e_field_probe import ElectricFieldProbe
from multipac_testbench.src.instruments.instrument import Instrument


STRING_TO_INSTRUMENT_CLASS = {
    'CurrentProbe': CurrentProbe,
    'ElectricFieldProbe': ElectricFieldProbe,
}  #:


class InstrumentFactory:
    """Class to create instruments."""

    def __init__(self):
        """Just instantiate I guess."""

    def run(self,
            name: str,
            df_data: pd.DataFrame,
            instrument_class_name: str,
            **instruments_kw: dict[str, Any],
            ) -> Instrument:
        """Take the proper subclass, instantiate it and return it.

        Parameters
        ----------
        name : str
            Name of the instrument, must correspond to a column in ``df_data``.
        df_data : pd.DataFrame
            Data of the multipactor test results.
        instrument_class_name : {'CurrentProbe', 'ElectricFieldProbe'}
            Name of the class, as given in the ``.toml`` file.
        instruments_kw : dict[str, Any]
            Other keyword arguments in the ``.toml`` file.

        Returns
        -------
        Instrument
            Instrument properly subclassed.

        """
        assert instrument_class_name in STRING_TO_INSTRUMENT_CLASS, \
            f"{instrument_class_name = } not recognized, check " \
            "STRING_TO_INSTRUMENT_CLASS in instrument/factory.py"

        instrument_class = STRING_TO_INSTRUMENT_CLASS[instrument_class_name]
        raw_data = df_data[name]
        return instrument_class(name, raw_data, **instruments_kw)
