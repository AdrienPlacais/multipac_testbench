#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store all the :class:`.InstrumentMultipactorBands` of a test."""
import numpy as np
from matplotlib.axes import Axes

from multipac_testbench.src.new_multipactor_band.instrument_multipactor_bands \
    import InstrumentMultipactorBands


class TestMultipactorBands(list):
    """Hold multipactor bands measured by all instruments during a test."""

    def __init__(
        self,
        instruments_multipactor_bands: list[InstrumentMultipactorBands | None]
    ) -> None:
        """Instantiate the object."""
        super().__init__(instruments_multipactor_bands)

    def plot_as_bool(self,
                     axes: Axes | None,
                     scale: float = 1.,
                     alpha: float = .5,
                     legend: bool = True,
                     **kwargs,
                     ) -> Axes:
        """Plot the multipactor bands."""
        original_scale = scale
        for instrument_multipactor_bands in self:
            if instrument_multipactor_bands is None:
                assert axes is not None
                axes.plot([], [])
                continue

            axes = instrument_multipactor_bands.plot_as_bool(axes=axes,
                                                             scale=scale,
                                                             alpha=alpha,
                                                             legend=legend,
                                                             **kwargs)
            scale += original_scale * 1e-2
        assert axes is not None
        return axes
