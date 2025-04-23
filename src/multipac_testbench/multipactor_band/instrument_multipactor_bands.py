"""Keep track of all multipactor bands measured by an :class:`.Instrument`."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from multipac_testbench.multipactor_band.creator import create
from multipac_testbench.multipactor_band.multipactor_band import (
    MultipactorBand,
)
from multipac_testbench.multipactor_band.polisher import POLISHER_T, polish
from numpy.typing import NDArray


class InstrumentMultipactorBands(list):
    """All :class:`.IMultipactorBand` of a test, by a given instrument."""

    def __init__(
        self,
        multipactor: NDArray[np.bool],
        power_is_growing: NDArray[np.bool],
        instrument_name: str,
        measurement_point_name: str,
        position: float,
        info_test: str = "",
        several_bands_politics: POLISHER_T = "merge",
        color: str | None = None,
    ) -> None:
        """Create the object.

        Parameters
        ----------
        multipactor :
            Array where True means multipactor, False no multipactor.
        power_is_growing :
            True where the power is growing, False where the power is
            decreasing, NaN where undetermined. The default is None, in which
            case it is not used.
        instrument_name :
            Name of the instrument that detected this multipactor.
        measurement_point_name :
            Where this multipactor was detected.
        position :
            Where multipactor was detected. If not applicable, in particular if
            the object represents multipactor anywhere in the testbench, it
            will be np.nan.
        info_test :
            TODO
        several_bands_politics :
            What to to when several multipactor bands are found in the same
            half power cycle:

            - ``'keep_lowest'``: we keep :class:`.MultipactorBand` at the
              lowest powers.
            - ``'keep_highest'``: we keep :class:`.MultipactorBand` at the
              highest powers.
            - ``'keep_all'``: we keep all :class:`.MultipactorBand`.
            - ``'merge'``: the resulting :class:`.MultipactorBand` will span
              from start of first :class:`.MultipactorBand` to end of last.
            - ``'keep_largest'``: we keep the :class:`.MultipactorBand` that
              was measured on the largest number of points.
        color :
            HTML color for plot, inherited from the :class:`.Instrument`.

        """
        bands = create(
            multipactor,
            power_is_growing,
            info=info_test + f" {instrument_name}",
        )
        bands = polish(bands, several_bands_politics)
        super().__init__(bands)

        self.multipactor = multipactor
        self.instrument_name = instrument_name
        self.measurement_point_name = measurement_point_name
        self.position = position
        self.color = color

        self._n_bands = len(self.actual_multipactor)

    def __str__(self) -> str:
        """Give concise information on the bands."""
        return self.instrument_name

    def __repr__(self) -> str:
        """Give information on how many bands were detected and how."""
        return f"{str(self)}: {self._n_bands} bands detected"

    def data_as_pd(self) -> pd.Series:
        """Return the multipactor data as a pandas Series."""
        ser = pd.Series(
            self.multipactor, name=f"MP detected by {self.instrument_name}"
        )
        return ser

    def plot_as_bool(
        self, axes: Axes | None = None, scale: float = 1.0, **kwargs
    ) -> Axes:
        """Plot as staircase like."""
        ser = self.data_as_pd().astype(float) * scale
        axes = ser.plot(ax=axes, color=self.color, **kwargs)
        assert axes is not None
        return axes

    @property
    def actual_multipactor(self) -> list[MultipactorBand]:
        """Filter out the :class:`.NoMultipactorBand`."""
        return [x for x in self if isinstance(x, MultipactorBand)]

    def lower_indexes(self) -> list[int | None]:
        """Get the indexes of all lower thresholds."""
        return [getattr(x, "lower_index", None) for x in self]

    def upper_indexes(self) -> list[int | None]:
        """Get the indexes of all upper thresholds."""
        return [getattr(x, "upper_index", None) for x in self]
