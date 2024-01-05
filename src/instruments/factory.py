#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to create the proper :class:`.Instrument`."""
from typing import Any

import pandas as pd

from multipac_testbench.src.instruments.current_probe import CurrentProbe
from multipac_testbench.src.instruments.e_field_probe import ElectricFieldProbe
from multipac_testbench.src.instruments.instrument import Instrument
from multipac_testbench.src.instruments.optical_fibre import OpticalFibre
from multipac_testbench.src.instruments.penning import Penning
from multipac_testbench.src.instruments.rf_power import RfPower


STRING_TO_INSTRUMENT_CLASS = {
    'CurrentProbe': CurrentProbe,
    'ElectricFieldProbe': ElectricFieldProbe,
    'OpticalFibre': OpticalFibre,
    'Penning': Penning,
    'RfPower': RfPower,
}  #:


class InstrumentFactory:
    """Class to create instruments."""

    def run(self,
            name: str,
            df_data: pd.DataFrame,
            class_name: str,
            **instruments_kw: Any,
            ) -> Instrument:
        """Take the proper subclass, instantiate it and return it.

        Parameters
        ----------
        name : str
            Name of the instrument, must correspond to a column in ``df_data``.
        df_data : pd.DataFrame
            Content of the multipactor tests results ``.csv`` file.
        class_name : {'CurrentProbe', 'ElectricFieldProbe', 'OpticalFibre',\
'Penning', 'RfPower'}
            Name of the instrument class, as given in the ``.toml`` file.
        instruments_kw : Any
            Other keyword arguments in the ``.toml`` file.

        Returns
        -------
        Instrument
            Instrument properly subclassed.

        """
        assert class_name in STRING_TO_INSTRUMENT_CLASS, \
            f"{class_name = } not recognized, check " \
            "STRING_TO_INSTRUMENT_CLASS in instrument/factory.py"

        instrument_class = STRING_TO_INSTRUMENT_CLASS[class_name]
        raw_data = df_data[name]
        return instrument_class(name, raw_data, **instruments_kw)
