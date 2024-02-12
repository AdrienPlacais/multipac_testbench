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


def _and(multipactor_in: list[np.ndarray[np.bool_]]) -> np.ndarray[np.bool_]:
    """Gather multipactor boolean arrays with the ``and`` operator.

    In other words: "Multipactor happens if all given instruments agree on it."

    """
    return np.array(multipactor_in).all(axis=0)


def _or(multipactor_in: list[np.ndarray[np.bool_]]) -> np.ndarray[np.bool_]:
    """Gather multipactor boolean arrays with the ``or`` operator.

    In other words: "Multipactor happens if one of the given instruments says
    that there is multipactor."

    """
    return np.array(multipactor_in).any(axis=0)


MULTIPACTOR_BANDS_MERGERS = {
    'strict': _and,
    'relaxed': _or,
}  #:


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

        self._n_bands = len(self)

    def __str__(self) -> str:
        """Give concise information on the bands."""
        return self.detector_instrument_name

    def __repr__(self) -> str:
        """Give information on how many bands were detected and how."""
        return f"{str(self)}: {self._n_bands} bands detected."

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

    @classmethod
    def from_other_multipactor_bands(cls,
                                     multiple_multipactor_bands: list[Self],
                                     union: str,
                                     name: str = '') -> Self:
        """Merge several :class:`MultipactorBands` objects.

        .. todo::
            Determine how and if transferring the list of
            :class:`.MultipactorBand` is useful.

        .. todo::
            Put a flag that will check consistency of position of MP bands.
            Like: ``assert_multipactor_bands_detected_at_same_position: bool``.

        Parameters
        ----------
        multipactor_bands : list[MultipactorBands]
            Objects to merge.
        union : {'strict', 'relaxed'}
            How the multipactor zones should be merged. It 'strict', all
            instruments must detect multipactor to consider that multipactor
            happened. If 'relaxed', only one instrument suffices.
        name : str, optional
            Name that will be given to the returned :class:`MultipactorBands`.
            The default is an empty string, in which case a default meaningful
            name will be given.

        Returns
        -------
        multipactor_bands : MultipactorBands

        """
        allowed = list(MULTIPACTOR_BANDS_MERGERS.keys())
        if union not in allowed:
            raise IOError(f"{union = }, while {allowed = }")

        dummy_multipactor_band = [0, 1, 2, 3]
        raise NotImplementedError("Determine how individual MultipactorBand "
                                  "objects should be transferred.")

        multipactor_in = [multipactor_band.multipactor
                          for multipactor_band in multiple_multipactor_bands]
        multipactor = MULTIPACTOR_BANDS_MERGERS[union](multipactor_in)

        if not name:
            name = f"{len(multiple_multipactor_bands)} instruments ({union})"

        multipactor_bands = cls(dummy_multipactor_band,
                                multipactor,
                                name,
                                )
        return multipactor_bands

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


if __name__ == '__main__':
    import pandas as pd

    multipac1 = np.array([True, True, True, False])
    multipac2 = np.array([True, True, False, False])
    multipac3 = np.array([True, False, False, False])
    multipac_in = [multipac1, multipac2, multipac3]

    df = pd.DataFrame({
        "MP1": multipac1,
        "MP2": multipac2,
        "MP3": multipac3,
        "strict": _and(multipac_in),
        "relaxed": _or(multipac_in)
    })
    print(df)
