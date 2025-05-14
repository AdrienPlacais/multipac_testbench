"""Define power probes to measure forward and reflected power."""

import numpy as np
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.util.filtering import (
    array_is_growing,
    remove_isolated_false,
    remove_trailing_true,
)
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
        """Identify regions where the signal is increasing ("growing").

        This method analyzes a signal to determine where it exhibits a growing
        trend. It returns a boolean array of the same length as the input
        signal, where ``True`` indicates a region of growth and ``False``
        otherwise.

        The method performs three main operations:
        1. It uses a sliding-window heuristic (*via* :func:`.array_is_growing`)
           to detect growth.
        2. It removes short, isolated ``False`` segments, enforcing a minimum
           number of consecutive ``True`` values to be considered valid.
        3. It clears any trailing ``True`` values near the end of the array to
           prevent spurious detections due to edge effects.

        Parameters
        ----------
        minimum_number_of_points :
            The minimum number of consecutive ``True`` values required to
            consider a region as growing. Shorter segments are suppressed.
        n_trailing_points_to_check :
            The number of points at the end of the signal to check and force to
            ``False`` if they form an isolated or uncertain growth pattern.
        **kwargs :
            Additional keyword arguments passed to :func:`.array_is_growing`.

        Returns
        -------
            Boolean array indicating where the signal is growing.

        Notes
        -----
        - The detection is influenced by the choice of parameters and the
          behavior of :func:`.array_is_growing`.
        - Trailing regions and short noise-like fluctuations are filtered out.

        .. todo::
           Consider adding post-processing to remove isolated ``True`` values.

        """
        n_points = len(self._raw_data)
        is_growing: list[bool] = []

        previous_value = True
        for i in range(n_points):
            local_is_growing = array_is_growing(
                self.data, i, undetermined_value=previous_value, **kwargs
            )

            is_growing.append(local_is_growing)
            previous_value = local_is_growing

        growth_mask = np.array(is_growing, dtype=np.bool_)

        # Remove isolated False
        if minimum_number_of_points > 0:
            growth_mask = remove_isolated_false(
                growth_mask, minimum_number_of_points
            )

        # Also ensure that last power growth is False
        if n_trailing_points_to_check > 0:
            growth_mask = remove_trailing_true(
                growth_mask,
                n_trailing_points_to_check,
                array_name_for_warning="power growth",
            )

        return growth_mask


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
