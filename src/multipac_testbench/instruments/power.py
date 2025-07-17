"""Define power probes to measure forward and reflected power."""

import logging
from functools import partial

import numpy as np
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.util.transfer_functions import power, power_channel_b
from multipac_testbench.util.types import POST_TREATER_T
from numpy.typing import NDArray


class Power(Instrument):
    """An instrument to measure power."""

    def __init__(
        self,
        *args,
        position: float = np.nan,
        a_calib: float | None = None,
        b_calib: float | None = None,
        k_fix: float | None = None,
        alpha_fix: float | None = None,
        **kwargs,
    ) -> None:
        """Instantiate the instrument, declare other specific attributes.

        See Also
        --------
        :func:`.transfer_functions.power`
        :func:`.transfer_functions.power_channel_b`

        Notes
        -----
        If ``k_fix`` and ``alpha_fix`` are provided, we add a second transfer
        function, :func:`.transfer_functions.power_channel_b`. It was proposed
        to fix the power measure on channel B.

        Parameters
        ----------
        a_calib :
            Calibration slope in :unit:`W/V`.
        b_calib :
            Calibration offset in :unit:`W`.
        k_fix :
            Fix slope constant.
        alpha_fix :
            Fix exponent constant.

        """
        self._a_calib: float
        if a_calib is not None:
            self._a_calib = a_calib
        self._b_calib: float
        if b_calib is not None:
            self._b_calib = b_calib
        self._a_fix: float
        if k_fix is not None:
            self._k_fix = k_fix
        self._alpha_fix: float
        if alpha_fix is not None:
            self._alpha_fix = alpha_fix
        super().__init__(*args, position=position, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power [W]"

    def where_is_growing(self, *args, **kwargs) -> NDArray[np.bool]:
        """Identify regions where the signal is increasing ("growing").

        .. deprecated:: 1.7.0
           Alias to :meth:`.Power.growth_mask`, consider calling it directly.

        """
        return self.growth_mask(*args, **kwargs)

    def growth_mask(
        self,
        minimum_number_of_points: int = 50,
        n_trailing_points_to_check: int = 40,
        width: int = 10,
        **kwargs,
    ) -> NDArray[np.bool]:
        return super().growth_mask(
            minimum_number_of_points=minimum_number_of_points,
            n_trailing_points_to_check=n_trailing_points_to_check,
            width=width,
            **kwargs,
        )

    @property
    def _transfer_functions(self) -> list[POST_TREATER_T]:
        assert hasattr(self, "_a_calib")
        assert hasattr(self, "_b_calib")

        funcs = [partial(power, a_calib=self._a_calib, b_calib=self._b_calib)]
        if hasattr(self, "_alpha_fix") and hasattr(self, "k_fix"):
            funcs.append(
                partial(
                    power_channel_b,
                    k_fix=self._k_fix,
                    alpha_fix=self._alpha_fix,
                )
            )
        return funcs


class ForwardPower(Power):
    """Store the forward power."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            self._is_raw
            and hasattr(self, "_alpha_fix")
            or hasattr(self, "k_fix")
        ):
            logging.warning(
                "ForwardPower typically measured on channel A, so you should "
                "not provide the arguments for the channel B fix."
            )

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Forward power $P_f$ [W]"


class ReflectedPower(Power):
    """Store the reflected power."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            self._is_raw
            and not hasattr(self, "_alpha_fix")
            or not hasattr(self, "k_fix")
        ):
            logging.warning(
                "ReflectedPower typically measured on channel B, so you should"
                " provide the arguments for the channel B fix."
            )

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Reflected power $P_r$ [W]"


class PowerSetpoint(Instrument):
    """Store the power asked by user.

    It should be preferred over :class:`.ForwardPower` to determine wether
    power is growing, as it is much more robust.

    Note
    ----
    Does not inherit from :class:`Power`.

    """

    def __init__(self, *args, position: float = np.nan, **kwargs) -> None:
        """Instantiate the instrument, declare other specific attributes."""
        super().__init__(*args, position=position, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power setpoint [dBm]"

    def growth_mask(
        self,
        minimum_number_of_points: int = 0,
        n_trailing_points_to_check: int = 0,
        width: int = 2,
        **kwargs,
    ) -> NDArray[np.bool]:
        return super().growth_mask(
            minimum_number_of_points=minimum_number_of_points,
            n_trailing_points_to_check=n_trailing_points_to_check,
            width=width,
            **kwargs,
        )

    @property
    def _transfer_functions(self) -> list[POST_TREATER_T]:
        return []
