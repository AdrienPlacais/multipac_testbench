"""Define current probe to measure multipactor cloud current."""

from functools import partial

from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.util.transfer_functions import current_probe
from multipac_testbench.util.types import POST_TREATER_T


class CurrentProbe(Instrument):
    """A probe to measure multipacting current."""

    def __init__(self, *args, a_probe: float | None = None, **kwargs) -> None:
        r"""Just instantiate.

        See Also
        --------
        :func:`.transfer_functions.current_probe`

        Parameters
        ----------
        a_probe :
            Calibration slope in :unit:`\\mu A/V`.

        """
        #: Calibration slope in :unit:`\\mu A/V`.
        self._a_probe: float
        if a_probe is not None:
            self._a_probe = a_probe

        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Multipactor current [$\mu$A]"

    @property
    def _transfer_functions(self) -> list[POST_TREATER_T]:
        """
        Give functions transforming acquisition voltage to physical quantity.

        They are used when input files contain raw data, ie acquisition
        voltages.

        """
        assert hasattr(self, "_a_probe")
        return [partial(current_probe, a_probe=self._a_probe)]
