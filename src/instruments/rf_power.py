#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define power probe to measure injected or reflected power."""
from multipac_testbench.src.instruments.instrument import Instrument


class RfPower(Instrument):
    """An instrument to measure injected of reflected rf power."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power [W]"
