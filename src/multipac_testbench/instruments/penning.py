"""Define Penning to measure evolution of pressure."""

from functools import partial
from typing import Self

import pandas as pd
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from multipac_testbench.util.physics import diff
from multipac_testbench.util.transfer_functions import pressure
from multipac_testbench.util.types import POST_TREATER_T


class Penning(Instrument):
    """A probe to measure pressure."""

    def __init__(
        self,
        *args,
        a_calib: float | None = None,
        b_calib: float | None = None,
        **kwargs,
    ) -> None:
        """Just instantiate.

        See Also
        --------
        :func:`.transfer_functions.pressure`

        Parameters
        ----------
        a_calib :
            Calibration slope in :unit:`1/V`.
        b_calib :
            Calibration offset.

        """
        #: Calibration slope in :unit:`1/V`.
        self._a_calib: float
        if a_calib is not None:
            self._a_calib = a_calib
        #: Calibration offset.
        self._b_calib: float
        if b_calib is not None:
            self._b_calib = b_calib
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "Pressure [mbar]"

    @property
    def _transfer_functions(self) -> list[POST_TREATER_T]:
        assert hasattr(self, "_a_calib")
        assert hasattr(self, "_b_calib")

        return [
            partial(pressure, a_calib=self._a_calib, b_calib=self._b_calib)
        ]


class DiffPenning(VirtualInstrument):
    """Store differential of pressure.

    This is sometimes easier to detect MP with this than with absolute pressure
    values.

    """

    def __init__(
        self, name: str, raw_data: pd.Series, penning: Penning, **kwargs
    ) -> None:
        """Create object, save :class:`.Power` objects."""
        super().__init__(name, raw_data, **kwargs)

        self._penning = penning
        self._penning.register_callback(self.recompute)

    def recompute(self) -> pd.Series:
        """Recompute pressure differential.

        This method is called when the stored :class:`Penning` attribute data
        is changed.

        """
        self._raw_data = diff(self._penning.data, name=self.name)
        self._notify_callbacks()
        return self._raw_data

    @classmethod
    def from_penning(
        cls, penning: Penning, name: str = "Diff_penning", **kwargs
    ) -> Self:
        return cls(
            name=name,
            raw_data=diff(penning.data, name),
            position=penning.position,
            penning=penning,
            color=penning.color,
            **kwargs,
        )

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Pressure differential $\delta P$ [mbar]"
