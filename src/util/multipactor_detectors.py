#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define functions to detect where multipactor happens."""
from typing import Any, Sequence
import numpy as np

from multipac_testbench.src.util.filtering import (remove_isolated_true,
                                                   remove_isolated_false)

def quantity_is_above_threshold(quantity: np.ndarray,
                                threshold: float,
                                consecutive_criterion: int = 0,
                                minimum_number_of_points: int = 1,
                                **kwargs: Any) -> np.ndarray:
    """Detect where ``quantity`` is above a given threshold.

    Parameters
    ----------
    quantity : np.ndarray
        Array of measured multipactor quantity.
    threshold : float
        Quantity value above which multipactor is detected.
    consecutive_criterion : int, optional
        If provided, we gather multipactor zones that were separated by
        ``consecutive_criterion`` measure points or less.
    minimum_number_of_points : int, optional
        If provided, the multipactor must happen on at least
        ``minimum_number_of_points`` consecutive points, otherwise we consider
        that it was a measurement flaw. The default is 1.


    Returns
    -------
    np.ndarray
        True where multipactor was detected.

    """
    multipactor = quantity >= threshold

    if consecutive_criterion > 0:
        multipactor = remove_isolated_false(multipactor, consecutive_criterion)

    if minimum_number_of_points > 1:
        multipactor = remove_isolated_true(multipactor, minimum_number_of_points)

    return multipactor


def start_and_end_of_contiguous_true_zones(
    multipactor: np.ndarray[np.bool_],
) -> list[tuple[int, int]]:
    """Get indexes of the entry and exit of contiguous multipactor zones.

    .. warning::
        ``starts`` is not the list of lower multipactor barrier indexes,
        ``ends`` is not the list of upper multipactor barrier indexes. To get
        this data, use :func:`indexes_of_lower_and_upper_multipactor_barriers`.

    Parameters
    ----------
    multipactor :  np.ndarray[np.bool_]
        Iterable where True means there is multipactor, False no multipactor,
        and np.NaN undetermined.

    Returns
    -------
    zones : Sequence[tuple[int, int]]
        List of first and last index of every multipactor band (multipactor
        contiguous zone).

    """
    diff = np.where(np.diff(multipactor))[0]
    n_changes = diff.size

    starts = (diff[::2] + 1).tolist()
    ends = (diff[1::2] + 1).tolist()

    # Multipacting zones are "closed"
    if n_changes % 2 == 0:
        # Multipacting zones are not closed
        if multipactor[0]:
            starts, ends = ends, starts
            starts.insert(0, 0)
            ends.append(None)

    # One multipacting zone is "open"
    else:
        ends.append(None)

        if multipactor[0]:
            starts, ends = ends, starts
            starts = ends
            starts.insert(0, 0)

    zones = [(start, end) for start, end in zip(starts, ends)]
    return zones
