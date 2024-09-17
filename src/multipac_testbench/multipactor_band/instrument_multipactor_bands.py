"""Keep track of all multipactor bands measured by an :class:`.Instrument`."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from multipac_testbench.multipactor_band.creator import create
from multipac_testbench.multipactor_band.multipactor_band import (
    MultipactorBand,
)
from multipac_testbench.multipactor_band.polisher import polish


class InstrumentMultipactorBands(list):
    """All :class:`IMultipactorBand` of a test, by a given instrument."""

    def __init__(
        self,
        multipactor: np.ndarray[np.bool_],
        power_is_growing: np.ndarray[np.bool_],
        instrument_name: str,
        measurement_point_name: str,
        position: float,
        info_test: str = "",
        several_bands_politics: str = "merge",
        color: str | None = None,
    ) -> None:
        """Create the object.

        Parameters
        ----------
        list_of_multipactor_band : list[IMultipactorBand]
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
            will be np.nan.
        power_is_growing : list[bool | float] | None, optional
            True where the power is growing, False where the power is
            decreasing, NaN where undetermined. The default is None, in which
            case it is not used.
        several_bands_politics : {'keep_highest', 'keep_lowest', 'keep_all',\
                'merge', 'keep_largest'}
            What to to when several multipactor bands are found in the same
            half power cycle:
                - ``'keep_lowest'``: we keep :class:`.MultipactorBand` at the
                lowest powers.
                - ``'keep_highest'``: we keep :class:`.MultipactorBand` at the
                highest powers.
                - ``'keep_all'``: we keep all :class:`.MultipactorBand`.
                - ``'merge'``: the resulting :class:`.MultipactorBand` will
                span from start of first :class:`.MultipactorBand` to end of
                last.
                - ``'keep_largest'``: we keep the :class:`.MultipactorBand`
                that was measured on the largest number of points.
        color : str | None, optional
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
