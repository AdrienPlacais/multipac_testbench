#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define power probe to measure power."""
from multipac_testbench.src.instruments.instrument import Instrument


class RfPower(Instrument):
    """An instrument to measure injected, reflected or probe rf power.

    .. deprecated:: v1.1.0

    """

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        print("RfPower.__init__ warning! Do not use this class anymore, will "
              "be deprecated in the future. Use Powers instead.")
        super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power [W]"