#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define field probe to measure electric field."""
from dataclasses import dataclass

from multipac_testbench.instruments.instrument import Instrument


@dataclass
class ElectricFieldProbe(Instrument):
    """A probe to measure electric field."""

    y_label: str = r"Voltage $[V]$"
