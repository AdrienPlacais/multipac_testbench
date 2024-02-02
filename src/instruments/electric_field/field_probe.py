#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define field probe to measure electric field."""
from multipac_testbench.src.instruments.electric_field.i_electric_field import\
    IElectricField


class FieldProbe(IElectricField):
    """A probe to measure electric field."""

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Measured voltage [V]"
