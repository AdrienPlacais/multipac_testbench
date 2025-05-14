"""Define power probes to measure forward and reflected power."""

import numpy as np
from multipac_testbench.instruments.instrument import Instrument
from numpy.typing import NDArray


class Power(Instrument):
    """An instrument to measure power."""

    def __init__(self, *args, position: float = np.nan, **kwargs) -> None:
        """Instantiate the instrument, declare other specific attributes."""
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
        **kwargs,
    ) -> NDArray[np.bool]:
        return super().growth_mask(
            minimum_number_of_points=minimum_number_of_points,
            n_trailing_points_to_check=n_trailing_points_to_check,
            **kwargs,
        )


class ForwardPower(Power):
    """Store the forward power."""

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Forward power $P_f$ [W]"


class ReflectedPower(Power):
    """Store the reflected power."""

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Reflected power $P_r$ [W]"
