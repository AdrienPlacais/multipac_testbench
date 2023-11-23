#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define field probe to measure electric field."""
import numpy as np
import pandas as pd

from multipac_testbench.instruments.instrument import Instrument


class ElectricFieldProbe(Instrument):
    """A probe to measure electric field."""

    def __init__(self,
                 name: str,
                 raw_data: pd.Series,
                 numerical_e_limits: np.ndarray | None = None,
                 analytical_e_limits: np.ndarray | None = None,
                 **kwargs,
                 ) -> None:
        """Instantiate the class."""
        super().__init__(name, raw_data, **kwargs)
        self.numerical_e_limits = numerical_e_limits
        self.analytical_e_limits = analytical_e_limits

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Voltage [V]"

    @property
    def mp_indexes(self) -> np.ndarray[np.int64]:
        """Determine index of measurements where MP should be detected."""
        raise NotImplementedError
        assert self.analytical_e_limits is not None, "You should give MP "\
            "limits to allow comparison."
        assert self.numerical_e_limits is not None, "You should give MP "\
            "limits to allow comparison."
        raise NotImplementedError("to do")
