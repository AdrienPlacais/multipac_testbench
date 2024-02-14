#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of all multipactor bands."""
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

    .. deprecated:: 1.4.0
        Same instance is stored in Instrument and IMeasurementPoint.

    """

    def __init__(self,
                 list_of_multipactor_band: list[MultipactorBand],
                 multipactor: np.ndarray[np.bool_],
                 instrument_name: str,
                 measurement_point_name: str,
                 position: float | None = None,
                 power_is_growing: list[bool | float] | None = None,
                 ) -> None:
        """Create the object.

        Parameters
        ----------
        list_of_multipactor_band : list[MultipactorBand]
            Individual multipactor bands.
        multipactor : np.ndarray[np.bool_]
            Array where True means multipactor, False no multipactor.
        instrument_name : str
            Name of the instrument that detected this multipactor.
        measurement_point_name : str
            Where this multipactor was detected.
        position : float
            Where multipactor was detected. If not applicable, in particular if
            the object represents multipactor anywhere in the testbench, it
            will be np.NaN.
        power_is_growing : list[bool | float] | None, optional
            True where the power is growing, False where the power is
            decreasing, NaN where undetermined. The default is None, in which
            case it is not used.

        """
        super().__init__(list_of_multipactor_band)
        self.multipactor = multipactor
        self.instrument_name = instrument_name
        self.measurement_point_name = measurement_point_name

        if position is None:
            print("MultipactorBands.__init__ warning: give it a position to "
                  "enable some consistency checkings.")
            position = 0.
        self.position = position

        self.power_is_growing: list[bool | float]
        if power_is_growing is not None:
            self.power_is_growing = power_is_growing

        self._n_bands = len(self)
        self._barriers: tuple[Sequence[int], Sequence[int]]

    def __str__(self) -> str:
        """Give concise information on the bands."""
        return self.instrument_name

    def __repr__(self) -> str:
        """Give information on how many bands were detected and how."""
        return f"{str(self)}: {self._n_bands} bands detected."

    @classmethod
    def from_ydata(
            cls,
            multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]],
            instrument_ydata: np.ndarray,
            instrument_name: str,
            measurement_point_name: str,
            position: float,
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
        instrument_name : str
            Name of the :class:`.Instrument`.
        measurement_point_name : str
            Name of the :class:`.IMeasurementPoint`.
        position : float
            Where multipactor was detected. If not applicable, in particular if
            the object represents multipactor anywhere in the testbench, it
            will be np.NaN.

        Returns
        -------
        MultipactorBands
            Instantiated object.

        """
        multipactor: np.ndarray[np.bool_]
        multipactor = multipac_detector(instrument_ydata)

        list_of_multipactor_band = \
            _multipactor_to_list_of_multipactor_band(multipactor,
                                                          instrument_name)

        multipactor_bands = cls(list_of_multipactor_band,
                                multipactor,
                                instrument_name,
                                measurement_point_name,
                                position)
        return multipactor_bands

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
        if not name:
            name = f"{len(multiple_multipactor_bands)} instruments ({union})"

        multipactor_in = [multipactor_band.multipactor
                          for multipactor_band in multiple_multipactor_bands]
        multipactor = MULTIPACTOR_BANDS_MERGERS[union](multipactor_in)
        list_of_multipactor_band = \
            _multipactor_to_list_of_multipactor_band(multipactor, name)

        positions = [mp_band.position
                     for mp_band in multiple_multipactor_bands]
        if len(set(positions)) == 1:
            position = positions[0]
        else:
            position = np.NaN

        multipactor_bands = cls(list_of_multipactor_band,
                                multipactor,
                                name,
                                name,
                                position=position,
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

        if hasattr(self, '_barriers'):
            return self._barriers

        barriers = indexes_of_lower_and_upper_multipactor_barriers(
            self.multipactor,
            self.power_is_growing)
        self._barriers = barriers
        self._tell_multipactor_band_if_it_reached_upper_threshold(barriers[1])
        return barriers

    def _tell_multipactor_band_if_it_reached_upper_threshold(
            self,
            upper_indexes: Sequence[int]
            ) -> None:
        """Save in :class:`MultipactorBand` objects if upper was reached."""
        for multipactor_band in self:
            if multipactor_band[0] in upper_indexes:
                multipactor_band.upper_threshold_was_reached = True
                continue
            if multipactor_band[-1] in upper_indexes:
                multipactor_band.upper_threshold_was_reached = True
                continue
            multipactor_band.upper_threshold_was_reached = False


def _multipactor_to_list_of_multipactor_band(
        multipactor: np.ndarray[np.bool_],
        instrument_name: str,
) -> list[MultipactorBand]:
    """Determine the contiguous multipacting zones."""
    starts_ends: list[tuple[int, int]]
    starts_ends = start_and_end_of_contiguous_true_zones(multipactor)

    list_of_multipactor_band = [
        MultipactorBand(start,
                        end,
                        instrument_name,
                        i)
        for i, (start, end) in enumerate(starts_ends)
    ]
    return list_of_multipactor_band


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
