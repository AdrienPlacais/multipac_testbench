#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define field probe to measure electric field."""
import pandas as pd

from multipac_testbench.src.instruments.instrument import Instrument


class ElectricFieldProbe(Instrument):
    """A probe to measure electric field."""

    def __init__(self,
                 name: str,
                 raw_data: pd.Series,
                 **kwargs,
                 ) -> None:
        """Instantiate the class."""
        super().__init__(name, raw_data, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Voltage [V]"
