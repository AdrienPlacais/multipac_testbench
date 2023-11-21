#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define field probe to measure electric field."""
from dataclasses import dataclass

import numpy as np

from multipac_testbench.instruments.instrument import Instrument


@dataclass
class ElectricFieldProbe(Instrument):
    """A probe to measure electric field."""

    y_label: str = r"Voltage $[V]$"
    numerical_e_limits: np.ndarray | None = None

    @property
    def mp_indexes(self) -> np.ndarray[np.int64]:
        """Determine index of measurements where MP should be detected."""
        assert self.numerical_e_limits is not None, "You should give MP "\
            "limits to allow comparison."
        raise NotImplementedError("to do")
