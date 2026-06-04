"""Define current probe to measure multipactor cloud current."""

import logging
from functools import partial
from typing import Literal, TypedDict

from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.util.transfer_functions import current_probe
from multipac_testbench.util.types import POST_TREATER_T


class AProbeCalibres(TypedDict):

    calibre_1mA: float
    calibre_10mA: float


class CurrentProbe(Instrument):
    """A probe to measure multipacting current."""

    def __init__(
        self, *args, a_probe: AProbeCalibres | float | None = None, **kwargs
    ) -> None:
        r"""Just instantiate.

        See Also
        --------
        :func:`.transfer_functions.current_probe`

        Parameters
        ----------
        a_probe :
            Calibration slope in :unit:`\\mu A/V`.

        """
        self.__a_probe: float
        self._a_probe_spec = a_probe
        if isinstance(a_probe, (float, int)):
            self._set_a_probe(a_probe=a_probe, calibre=None)
        return super().__init__(*args, **kwargs)

    @property
    def _a_probe(self) -> float:
        """Calibration slope in :unit:`\\mu A/V`."""
        return self.__a_probe

    @_a_probe.setter
    def _a_probe(self, value: float) -> None:
        """Clean the cached data at each update of the calibration constant."""
        self.__a_probe = value
        for attr in ("_data", "_data_as_pd"):
            if hasattr(self, attr):
                delattr(self, attr)

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
        if not hasattr(self, "_a_probe"):
            return []
        return [partial(current_probe, a_probe=self._a_probe)]

    def _set_a_probe(
        self,
        a_probe: AProbeCalibres | float | None = None,
        calibre: Literal[1, 10] | None = None,
    ) -> None:
        """Set the appropriate calibration constant."""
        if isinstance(a_probe, (float, int)):
            self._a_probe = a_probe
            return

        if a_probe is None:
            logging.error("Provided `a_probe` is None.")
            return

        if calibre is None:
            logging.error(
                "You must either provide the `a_probe` directly, either "
                "provide a dictionary with `'calibre_1mA'` and "
                "`'calibre_10mA'` keys along with the actual calibre value:"
                "`1` or `10`."
            )
            return

        if calibre == 1:
            self._a_probe = a_probe["calibre_1mA"]
            return
        if calibre == 10:
            self._a_probe = a_probe["calibre_10mA"]
            return

        logging.error(f"Error in arguments: {a_probe = }, {calibre = }")
