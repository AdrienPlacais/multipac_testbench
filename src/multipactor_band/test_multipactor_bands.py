#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store all the :class:`.InstrumentMultipactorBands` of a test."""
import numpy as np
from matplotlib.axes import Axes

from multipac_testbench.src.multipactor_band.instrument_multipactor_bands \
    import InstrumentMultipactorBands


class TestMultipactorBands(list):
    """Hold multipactor bands measured by all instruments during a test."""

    def __init__(
        self,
        instruments_multipactor_bands: list[InstrumentMultipactorBands | None],
        power_is_growing: np.ndarray[np.bool_],
    ) -> None:
        """Instantiate the object."""
        super().__init__(instruments_multipactor_bands)
        self.power_is_growing = power_is_growing

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

    def merge(self,
              union: str,
              name: str = '',
              filter_out_none: bool = True,
              info_test: str = '',
              several_bands_politics: str = 'merge',
              ) -> InstrumentMultipactorBands:
        """Merge the :class:`InstrumentMultipactorBands` in ``self``.

        For that, we merge their ``multipactor`` boolean numpy array and
        recreate a :class:`.InstrumentMultipactorBands` with its own
        :class:`.MultipactorBand`.

        .. todo::
            Put a flag that will check consistency of position of MP bands.
            Like: ``assert_instrument_multipactor_bands_detected_at_same_posit\
ion: bool``.

        Parameters
        ----------
        union : {'strict', 'relaxed'}
            How the multipactor zones should be merged. It 'strict', all
            instruments must detect multipactor to consider that multipactor
            happened. If 'relaxed', only one instrument suffices.
        name : str, optional
            Name that will be given to the returned
            :class:`InstrumentMultipactorBands`. The default is an empty
            string, in which case a default meaningful name will be given.
        filter_out_none : bool, optional
            To remove the ``None`` in ``self``. The default is True.
        info_test : str, optional
            To give more explicit output when there is a problem in the merging
            process. The default is an empty string.
        several_bands_politics : {'keep_first', 'keep_last', 'keep_all', \
                'merge'}
            What to to when several multipactor bands are found in the same
            half-power cycle:
                - ``'keep_first'``: we keep first :class:`.MultipactorBand`
                - ``'keep_last'``: we keep last :class:`.MultipactorBand`
                - ``'keep_all'``: we keep all :class:`.MultipactorBand`
                (currently not implemented)
                - ``'merge'``: the final :class:`.MultipactorBand` spans from
                start of first :class:`.MultipactorBand` to end of last.


        Returns
        -------
        instrument_multipactor_bands : InstrumentMultipactorBands
            Object holding merged multipactor bands.

        """
        if not filter_out_none:
            raise NotImplementedError
        filtered = [band for band in self if band is not None]

        allowed = list(MULTIPACTOR_ARRAY_MERGERS.keys())
        if union not in allowed:
            raise IOError(f"{union = }, while {allowed = }")
        multipactor_in = [band.multipactor for band in filtered]
        multipactor = MULTIPACTOR_ARRAY_MERGERS[union](multipactor_in)

        if not name:
            name = f"{len(self)} instruments ({union})"

        positions = [band.position for band in filtered]
        if len(set(positions)) == 1:
            position = positions[0]
        else:
            position = np.NaN

        instrument_multipactor_bands = InstrumentMultipactorBands(
            multipactor,
            self.power_is_growing,
            instrument_name=name,
            measurement_point_name=name,
            position=position,
            info_test=info_test,
            several_bands_politics=several_bands_politics,
        )
        return instrument_multipactor_bands


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


MULTIPACTOR_ARRAY_MERGERS = {
    'strict': _and,
    'relaxed': _or,
}  #:
