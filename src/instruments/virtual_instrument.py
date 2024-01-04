#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an instrument-like object."""
from collections.abc import Iterable
from typing import Self

import numpy as np
import pandas as pd

from multipac_testbench.src.instruments.instrument import Instrument


class VirtualInstrument(Instrument):
    """An object that works like an :class:`.Instrument`.

    Allows to avoid confusion when the object under study should have same
    methods than a classic instrument, but is user-defined with analytical data
    or data calculated from other instruments.

    """

    def __init__(self,
                 name: str,
                 raw_data: pd.Series,
                 **kwargs) -> None:
        """Instantiate object."""
        super().__init__(name, raw_data, **kwargs)

    @classmethod
    def from_array(cls,
                   name: str,
                   ydata: np.ndarray,
                   xdata: Iterable | None = None,
                   **kwargs) -> Self:
        """Instantiate from numpy array."""
        if xdata is None:
            n_points = len(ydata)
            xdata = range(1, n_points + 1)

        raw_data = pd.Series(data=ydata,
                             index=xdata,
                             name=name)

        return cls(name, raw_data, **kwargs)
