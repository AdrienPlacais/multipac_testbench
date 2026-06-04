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
        super().__init__(*args, **kwargs)
        if isinstance(a_probe, (float, int)):
            self._set_a_probe(calibre=None)

    @property
    def _a_probe(self) -> float:
        """Calibration slope in :unit:`\\mu A/V`."""
        return self.__a_probe

    @_a_probe.setter
    def _a_probe(self, value: float) -> None:
        """Clean the cached data at each update of the calibration constant."""
        self.__a_probe = value
        self._post_treaters = [
            pt
            for pt in self._post_treaters
            if not (isinstance(pt, partial) and pt.func is current_probe)
        ]
        self._post_treaters.append(partial(current_probe, a_probe=value))
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
        if not hasattr(self, "__a_probe"):
            return []
        return [partial(current_probe, a_probe=self._a_probe)]

    def _set_a_probe(self, calibre: Literal[1, 10] | None = None) -> None:
        """Set the appropriate calibration constant."""
        if isinstance(self._a_probe_spec, (float, int)):
            self._a_probe = self._a_probe_spec
            return

        if self._a_probe_spec is None:
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
            self._a_probe = self._a_probe_spec["calibre_1mA"]
            return
        if calibre == 10:
            self._a_probe = self._a_probe_spec["calibre_10mA"]
            return

        logging.error(
            f"Error in arguments: {self._a_probe_spec = }, {calibre = }"
        )
