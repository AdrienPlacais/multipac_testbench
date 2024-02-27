#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of all multipactor bands.

.. todo::
    Maybe this class should also hold a None or something when multipactor did
    not appear during a power cycle.
    If there is multipactor at the start of the test but it is conditioned,
    this information does not appear!!

"""
from collections.abc import Callable
from typing import Self

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from multipac_testbench.src.new_multipactor_band.multipactor_band import \
    MultipactorBand


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
    """All :class:`MultipactorBand` of a test, at a given localisation."""

    def __init__(self,
                 multipactor: np.ndarray[np.bool_],
                 power_is_growing: np.ndarray[np.bool_],
                 instrument_name: str,
                 measurement_point_name: str,
                 position: float,
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
        list_of_multipactor_band = multipactor_to_list_of_mp_band(
            multipactor,
            power_is_growing,
        )
        super().__init__(list_of_multipactor_band)
        self.multipactor = multipactor

        self.instrument_name = instrument_name
        self.measurement_point_name = measurement_point_name
        self.position = position

        self._n_bands = len([x for x in self if x is not None])

    def __str__(self) -> str:
        """Give concise information on the bands."""
        return self.instrument_name

    def __repr__(self) -> str:
        """Give information on how many bands were detected and how."""
        return f"{str(self)}: {self._n_bands} bands detected"

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
        raise NotImplementedError
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

    def data_as_pd(self) -> pd.Series:
        """Return the multipactor data as a pandas Series."""
        ser = pd.Series(self.multipactor,
                        name=f"MP detected by {self.instrument_name}")
        return ser

    def plot_as_bool(self,
                     axes: Axes | None = None,
                     scale: float = 1.,
                     **kwargs
                     ) -> Axes:
        """Plot as staircase like."""
        ser = self.data_as_pd().astype(float) * scale
        axes = ser.plot(ax=axes, **kwargs)
        return axes

    def lower_indexes(self) -> list[int | None]:
        """Get the indexes of all lower thresholds."""
        return [x.lower_index if x is not None else None
                for x in self]

    def upper_indexes(self) -> list[int | None]:
        """Get the indexes of all upper thresholds."""
        return [x.upper_index if x is not None else None for x in self]

    def first_indexes(self) -> list[int | None]:
        """Get the indexes of entry of every zone."""
        return [x.first_index if x is not None else None for x in self]

    def last_indexes(self) -> list[int | None]:
        """Get the indexes of exit of every zone."""
        return [x.last_index if x is not None else None for x in self]


def multipactor_to_list_of_mp_band(multipactor: np.ndarray[np.bool_],
                                   power_is_growing: np.ndarray[np.bool_],
                                   ) -> list[MultipactorBand | None]:
    """Create the different :class:`MultipactorBand`.

    Parameters
    ----------
    multipactor : np.ndarray[np.bool_]
        True means multipactor, False no multipactor.
    power_is_growing : np.ndarray[float]
        True means power is growing, False it is decreasing.

    Returns
    -------
    list[MultipactorBand | None]
        One object per half power cycle (i.e. one object for power growth, one
        for power decrease). None means that no multipactor was detected.

    """
    delta_multipactor = np.diff(multipactor)
    delta_power_is_growing = np.diff(power_is_growing)
    zip_enum = enumerate(zip(delta_multipactor, delta_power_is_growing))

    all_bands: list[MultipactorBand | None] = []
    current_band: None | MultipactorBand = None
    first_index: None | int = None
    last_index: None | int = None
    for i, (change_in_multipactor, change_in_power_growth) in zip_enum:
        # we attack a new power cycle, or the second half of previous one
        if change_in_power_growth:
            # specific case: did not go out of the MP band
            if first_index is not None and last_index is None:
                assert current_band is None
                reached_second_threshold = False
                last_index = i
                current_band = MultipactorBand(first_index,
                                               last_index,
                                               reached_second_threshold,
                                               power_is_growing[i])

                # reset for next cycle
                first_index = i + 1
            else:
                first_index = None

            # we finish with this power cycle
            all_bands.append(current_band)
            current_band = None
            last_index = None
            # idx_end is always None
            # but idx_start is set to current step if we did not manage to exit
            # the last multipactor band
            continue

        # we are continuing a power cycle
        if not change_in_multipactor:
            # we are already multipacting or already not multipacting
            continue

        # there is entry or exit of a mp zone
        # entry of a new mp zone
        if multipactor[i + 1]:
            first_index = i + 1
            continue

        # exit of a mp zone
        assert first_index is not None
        last_index = i
        reached_second_threshold = True
        if current_band is not None:
            print("MultipactorBands warning: I guess there was two MP bands "
                  "for this power cycle!! To investigate. Only keeping second "
                  "one...")
        current_band = MultipactorBand(first_index, last_index,
                                       reached_second_threshold,
                                       power_is_growing[i])
    return all_bands
