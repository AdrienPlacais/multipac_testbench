#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of a single multipactor band.

i.e.: a set of measurement points where multipactor happens. A
:class:`MultipactorBand` is defined on half a power cycle.

"""
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
        assert first_index is not None, ("We are exiting a multipacting zone "
                                         "but I did not detect when it started"
                                         ". Check what happened around index "
                                         f"{i}.")
        last_index = i
        reached_second_threshold = True
        if current_band is not None:
            print("MultipactorBand warning: I guess there was two MP bands "
                  "for this power cycle!! To investigate. Only keeping second "
                  "one...")
        current_band = MultipactorBand(first_index, last_index,
                                       reached_second_threshold,
                                       power_is_growing[i])
    return all_bands
