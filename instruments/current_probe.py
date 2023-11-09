#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define current probe to measure multipactor cloud current."""
from dataclasses import dataclass

from multipac_testbench.instruments.instrument import Instrument


@dataclass
class CurrentProbe(Instrument):
    """A probe to measure multipacting current."""

    y_label: str = r"MP current $[\mu A]$"
