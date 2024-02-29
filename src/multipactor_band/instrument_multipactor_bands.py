#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of all multipactor bands measured by an :class:`.Instrument`."""
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from multipac_testbench.src.multipactor_band.multipactor_band import \
    multipactor_to_list_of_mp_band


class InstrumentMultipactorBands(list):
    """All :class:`MultipactorBand` of a test, by a given instrument."""

    def __init__(self,
                 multipactor: np.ndarray[np.bool_],
                 power_is_growing: np.ndarray[np.bool_],
                 instrument_name: str,
                 measurement_point_name: str,
                 position: float,
                 info_test: str = ''
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
            info=info_test + f" {instrument_name}",
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
