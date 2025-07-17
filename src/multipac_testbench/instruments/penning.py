"""Define Penning to measure evolution of pressure."""

from functools import partial

from multipac_testbench.instruments.instrument import Instrument
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
