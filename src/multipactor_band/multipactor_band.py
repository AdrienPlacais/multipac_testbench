#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of a single multipactor band.

i.e.: a set of measurement points where multipactor happens. A
:class:`MultipactorBand` is defined on half a power cycle.

"""
import logging
import numpy as np


class MultipactorBand:
    """First and last index of a multipactor zone."""

    def __init__(self,
                 first_index: int,
                 last_index: int,
                 reached_second_threshold: bool,
                 power_grows: bool) -> None:
        """Instantiate object."""
        self.first_index = first_index
        self.last_index = last_index

        self.lower_index, self.upper_index = first_index, last_index
        if not power_grows:
            self.upper_index, self.lower_index = first_index, last_index

        if not reached_second_threshold:
            self.upper_index = None
        self.indexes = [i for i in range(self.first_index,
                                         self.last_index + 1)]
        self.reached_second_threshold = reached_second_threshold

    def __repr__(self) -> str:
        """Inform."""
        out = f"{self.first_index = }, {self.last_index = }, "
        out += f"{self.lower_index = }, {self.upper_index = }"
        return out


def _end_half_power_cycle(
        first_index: int | None,
        last_index: int | None,
        index: int,
        current_band: MultipactorBand | None,
        power_grows: bool,
) -> tuple[int | None, MultipactorBand | None]:
    """Start a new power cycle."""
    # Did not go out of the multipactor band
    if first_index is not None and last_index is None:
        assert current_band is None
        last_index = index
        current_band = MultipactorBand(first_index,
                                       last_index,
                                       reached_second_threshold=False,
                                       power_grows=power_grows)

        first_index = index + 1
        return first_index, current_band

    return first_index, current_band


def _init_half_power_cycle(first_index: int | None = None
                           ) -> tuple[int | None, None, None]:
    """(Re)-init variables for a new half power cycle."""
    last_index = None
    current_band = None
    return first_index, last_index, current_band


def _enter_a_mp_zone(last_index: int | None,
                     index: int,
                     current_band: MultipactorBand | None,
                     info: str,
                     ) -> int:
    """Enter a multipactor zone."""
    first_index = index + 1
    if last_index is not None:
        pass
    if current_band is not None:
        logging.debug(
            f"{info} entering a new MP zone, but it is not the first of the "
            "half-power cycle.")
    return first_index


def _exit_a_mp_zone(first_index: int | None,
                    index: int,
                    current_band: MultipactorBand | None,
                    power_grows: bool,
                    info: str,
                    ) -> tuple[int, MultipactorBand]:
    """Exit a multipactor zone."""
    assert first_index is not None, (
        f"{info}: we are exiting a multipacting zone but I did not detect "
        f"when it started. Check what happened around {index = }.")

    last_index = index

    if current_band is not None:
        logging.warning(
            f"{info}: detected two multipactor bands in the same half-"
            "power cycle. First one spanned from index "
            f"{current_band.first_index} to {current_band.last_index}. "
            f"Second one from {first_index} to {last_index}. I will merge "
            "them.")
        first_index = current_band.first_index

    current_band = MultipactorBand(first_index,
                                   last_index,
                                   reached_second_threshold=True,
                                   power_grows=power_grows)
    return last_index, current_band


def multipactor_to_list_of_mp_band(multipactor: np.ndarray[np.bool_],
                                   power_is_growing: np.ndarray[np.bool_],
                                   info: str = '',
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

    all_bands = []
    first_index, last_index, current_band = _init_half_power_cycle()

    for i, (change_in_multipactor, change_in_power_growth) in zip_enum:
        if not (change_in_multipactor or change_in_power_growth):
            continue

        if change_in_power_growth:
            first_index, current_band = _end_half_power_cycle(
                first_index,
                last_index,
                i,
                current_band,
                bool(power_is_growing[i])
            )

            all_bands.append(current_band)
            first_index, last_index, current_band = _init_half_power_cycle(
                first_index)
            continue

        if multipactor[i + 1]:
            first_index = _enter_a_mp_zone(last_index,
                                           i,
                                           current_band,
                                           info)
            continue

        last_index, current_band = _exit_a_mp_zone(first_index,
                                                   i,
                                                   current_band,
                                                   bool(power_is_growing[i]),
                                                   info)
    return all_bands
