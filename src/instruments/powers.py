#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Define power probe to measure forward and reflected power.

Also include calculation of reflection coefficient :math:`\Gamma` and (voltage)
SWR :math:`SWR`.

"""
import pandas as pd

from multipac_testbench.src.instruments.instrument import Instrument


class Powers(Instrument):
    """An instrument to measure forward and reflected powers."""

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power [W]"
