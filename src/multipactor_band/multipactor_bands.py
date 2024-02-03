#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of all multipactor bands, at a given position."""
from collections.abc import Callable
from typing import Self

import numpy as np

from multipac_testbench.src.multipactor_band.multipactor_band import \
    MultipactorBand
from multipac_testbench.src.util.multipactor_detectors import (
    indexes_of_lower_and_upper_multipactor_barriers,
    start_and_end_of_contiguous_true_zones
)


class MultipactorBands(list):
    """All :class:`MultipactorBand` of a test, at a given localisation."""

    def __init__(self, multipactor_bands: list[MultipactorBand]) -> None:
        """Create the object."""
        super().__init__(multipactor_bands)

    @classmethod
    def from_ydata(cls,
                   multipac_detector: Callable,
                   instrument_ydata: np.ndarray,
                   power_is_growing: list[bool | float],
                   detector_instrument_name: str,
                   ) -> Self:
        """
        Detect where multipactor happens, create all :class:`MultipactorBand`.

        .. todo::
            Maybe add the name of the instrument somewhere?

        """
        multipactor: list[bool | float]
        multipactor = multipac_detector(instrument_ydata)

        starts_ends: list[tuple[int, int]]
        starts_ends = start_and_end_of_contiguous_true_zones(multipactor)

        # lower_upper = indexes_of_lower_and_upper_multipactor_barriers(
        #     multipactor,
        #     power_is_growing
        # )

        multipactor_bands = [
            MultipactorBand(start, end, detector_instrument_name)
            for start, end in starts_ends
        ]

        return cls(multipactor_bands)
