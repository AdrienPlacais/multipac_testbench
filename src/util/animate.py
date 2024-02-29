#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define helper functions for the animation plots."""
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes

import multipac_testbench.src.instruments as ins


def get_limits(
        axes_instruments: dict[Axes, Sequence[ins.Instrument]],
        instruments_to_ignore_for_limits: Sequence[ins.Instrument | str] = (),
) -> dict[Axes, tuple[float, float]]:
    """Define constant limits for the animations.

    Parameters
    ----------
    axes_instruments : dict[Axes, Sequence[ins.Instrument]]
        Dictionary linking all the :class:`ins.Instrument` to the Axe they
        should be plotted onto.
    instruments_to_ignore_for_limits : Sequence[ins.Instrument | str]
        Instruments that should not modify the limits.

    Returns
    -------
    dict[Axes, tuple[float, float]]
        Dictionary linking avery Axe with its limits.

    """
    names_to_ignore = [x if isinstance(x, str) else x.name
                       for x in instruments_to_ignore_for_limits]
    limits = {}
    for axe, instruments in axes_instruments.items():
        all_data = [instrument.data for instrument in instruments
                     if instrument.name not in names_to_ignore]

        lowers = [np.nanmin(data) for data in all_data]
        lower = min(lowers)

        uppers = [np.nanmax(data) for data in all_data]
        upper = max(uppers)
        amplitude = abs(upper - lower)

        limits[axe] = (lower - .1 * amplitude, upper + .1 * amplitude)
    return limits
