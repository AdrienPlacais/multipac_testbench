#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a fake frequency probe."""
from typing import Self
import numpy as np
import pandas as pd

from multipac_testbench.src.instruments.virtual_instrument import \
    VirtualInstrument


class Frequency(VirtualInstrument):
    r"""Store a frequency.

    By default, the frequency is in :math:`\mathrm{MHz}`.

    """

    @classmethod
    def from_user_defined_frequency(cls,
                                    freq_mhz: float,
                                    n_points: int,
                                    name: str = 'Reference frequency',
                                    **kwargs
                                    ) -> Self:
        r"""Instantiate the object with a constant frequency.

        Parameters
        ----------
        freq_mhz : float
            Frequency in :math:`\mathrm{MHz}`.
        n_points : int
            Number of points to fill.
        name : str
            Name of the series and of the instrument.
        kwargs :
            Other keyword arguments passed to pd.Series and constructor.

        Returns
        -------
        Self

        """
        raw_data = np.full(n_points, freq_mhz)
        df_data = pd.Series(raw_data, name=name, **kwargs)
        return cls(name, df_data, position=np.NaN, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"RF frequency $f~[\mathrm{MHz}]$"
