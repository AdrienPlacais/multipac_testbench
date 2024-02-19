#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to create the proper :class:`.Instrument`."""
from typing import Any

import pandas as pd

from multipac_testbench.src.instruments.current_probe import CurrentProbe
from multipac_testbench.src.instruments.electric_field.field_probe import \
    FieldProbe
from multipac_testbench.src.instruments.instrument import Instrument
from multipac_testbench.src.instruments.optical_fibre import OpticalFibre
from multipac_testbench.src.instruments.penning import Penning
from multipac_testbench.src.instruments.powers import Powers


STRING_TO_INSTRUMENT_CLASS = {
    'CurrentProbe': CurrentProbe,
    'ElectricFieldProbe': FieldProbe,
    'FieldProbe': FieldProbe,
    'OpticalFibre': OpticalFibre,
    'Penning': Penning,
    'Powers': Powers,
}  #:


class InstrumentFactory:
    """Class to create instruments."""

    def run(self,
            name: str,
            df_data: pd.DataFrame,
            class_name: str,
            column_header: str | list[str] | None = None,
            **instruments_kw: Any,
            ) -> Instrument:
        """Take the proper subclass, instantiate it and return it.

        Parameters
        ----------
        name : str
            Name of the instrument. For clarity, it should match the name of a
            column in ``df_data`` when it is possible.
        df_data : pd.DataFrame
            Content of the multipactor tests results ``.csv`` file.
        class_name : {'CurrentProbe', 'ElectricFieldProbe', 'OpticalFibre',\
'Penning', 'Power'}
            Name of the instrument class, as given in the ``.toml`` file.
        column_header : str | list[str] | None, optional
            Name of the column(s) from which the ydata of the instrument will
            be taken. The default is None, in which case ``column_header`` is
            set to ``name``. In general it is not necessary to provide it. An
            exception is when several ``.csv`` columns should be loaded in the
            instrument.
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

        if column_header is None:
            column_header = name

        raw_data = df_data[column_header]

        if isinstance(raw_data, pd.DataFrame):
            return instrument_class.from_pd_dataframe(name,
                                                      raw_data,
                                                      **instruments_kw)
        return instrument_class(name,
                                raw_data,
                                **instruments_kw)
