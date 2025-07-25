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
        p_low: float | None = None,
        p_high: float | None = None,
        k_fix: float | None = None,
        alpha_fix: float | None = None,
        ensure_no_negative: bool = True,
        **kwargs,
    ) -> None:
        r"""Instantiate the instrument, declare other specific attributes.

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
        p_low, p_high :
            Lowest and highest measurable powers in :unit:`W/V`. Must
            correspond to what is set in the watt meter. Correspond to
            ``REC_LIM_LOW`` and ``REC_LIM_UPP`` in LabView.
        k_fix :
            Fix slope constant.
        alpha_fix :
            Fix exponent constant.
        ensure_no_negative :
            Set negative powers to :math:`0~\mathrm{V}`. Should be useless.

        """
        self._a_calib: float
        self._b_calib: float
        self._ensure_no_negative = ensure_no_negative
        if p_low is not None and p_high is not None:
            self._a_calib, self._b_calib = self._get_wattmeter_calibration(
                p_low, p_high
            )

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

    def _get_wattmeter_calibration(
        self, p_low: float, p_high: float, v_low: float = 0.0, v_high=1.0
    ) -> tuple[float, float]:
        r"""Compute the wattmetre transfer function parameters.

        We just find the linear relation parameters:

        .. math:
            P_\mathrm{W} = a_\mathrm{calib} \times V_\mathrm{acqui} +
            b_\mathrm{calib}

        """
        a_calib = (p_high - p_low) / (v_high - v_low)
        b_calib = p_low - a_calib * v_low
        return a_calib, b_calib

    @property
    def _transfer_functions(self) -> list[POST_TREATER_T]:
        assert hasattr(self, "_a_calib")
        assert hasattr(self, "_b_calib")

        funcs = [
            partial(
                power,
                a_calib=self._a_calib,
                b_calib=self._b_calib,
                ensure_no_negative=self._ensure_no_negative,
            )
        ]
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
            or hasattr(self, "_k_fix")
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
        if self._is_raw and not (
            hasattr(self, "_alpha_fix") and hasattr(self, "_k_fix")
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
