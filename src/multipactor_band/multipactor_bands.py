#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of all multipactor bands, at a given position."""
from collections.abc import Callable, Sequence
from typing import Self

import numpy as np

from multipac_testbench.src.multipactor_band.multipactor_band import \
    MultipactorBand
from multipac_testbench.src.util.multipactor_detectors import (
    indexes_of_lower_and_upper_multipactor_barriers,
    start_and_end_of_contiguous_true_zones
)


class MultipactorBands(list):
    """All :class:`MultipactorBand` of a test, at a given localisation.

    .. todo::
        Same instance is stored in Instrument and IMeasurementPoint.

    """

    def __init__(self,
                 multipactor_bands: list[MultipactorBand],
                 multipactor: np.ndarray[np.bool_],
                 detector_instrument_name: str,
                 power_is_growing: list[bool | float] | None = None,
                 ) -> None:
        """Create the object.

        Parameters
        ----------
        multipactor_bands : list[MultipactorBand]
            Individual multipactor bands.
        multipactor : np.ndarray[np.bool_]
            Array where True means multipactor, False no multipactor.
        detector_instrument_name : str
            Name of the instrument that detected multipactor.
        power_is_growing : list[bool | float] | None, optional
            True where the power is growing, False where the power is
            decreasing, NaN where undetermined. The default is None, in which
            case it is not used.

        """
        super().__init__(multipactor_bands)
        self.multipactor = multipactor
        self.detector_instrument_name = detector_instrument_name

        self.power_is_growing: list[bool | float]
        if power_is_growing is not None:
            self.power_is_growing = power_is_growing

    def __str__(self) -> str:
        """Give concise information on the bands."""
        return self.detector_instrument_name

    @classmethod
    def from_ydata(
            cls,
            multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]],
            instrument_ydata: np.ndarray,
            detector_instrument_name: str,
            ) -> Self:
        """Detect where multipactor happens, create :class:`MultipactorBand`.

        Parameters
        ----------
        multipac_detector : Callable[[np.ndarray], np.ndarray[np.bool_]]
            Function taking in an :class:`.Instrument` ``ydata``, returning a
            boolean array where True means multipactor and False no
            multipactor.
        instrument_ydata : np.ndarray
            The ``ydata`` from the :class:`.Instrument`.
        detector_instrument_name : str
            Name of the :class:`.Instrument`.

        Returns
        -------
        MultipactorBands
            Instantiated object.

        """
        multipactor: np.ndarray[np.bool_]
        multipactor = multipac_detector(instrument_ydata)

        starts_ends: list[tuple[int, int]]
        starts_ends = start_and_end_of_contiguous_true_zones(multipactor)

        multipactor_bands = [
            MultipactorBand(start, end, detector_instrument_name)
            for start, end in starts_ends
        ]

        return cls(multipactor_bands, multipactor, detector_instrument_name)

    @property
    def barriers(self) -> tuple[Sequence[int], Sequence[int]]:
        """Get list of indexes of lower and upper barriers.

        Returns
        -------
        lower_indexes : Sequence[int]
            Indexes corresponding to a crossing of lower multipactor barrier.
        upper_indexes : Sequence[int]
            Indexes corresponding to a crossing of upper multipactor barrier.

        """
        assert hasattr(self, 'power_is_growing'), (
            "You need to set MultipactorBands.power_is_growing to discriminate"
            "lower threshold from upper threshold.")
        barriers = indexes_of_lower_and_upper_multipactor_barriers(
            self.multipactor,
            self.power_is_growing)
        return barriers
