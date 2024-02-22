#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define power probes to measure forward and reflected power."""
import numpy as np

from multipac_testbench.src.instruments.instrument import Instrument
from multipac_testbench.src.util.filtering import (
    array_is_growing,
    remove_trailing_true,
    remove_isolated_false,
)


class Power(Instrument):
    """An instrument to measure power."""

    def __init__(self, *args, position: float = np.NaN, **kwargs) -> None:
        """Instantiate the instrument, declare other specific attributes."""
        super().__init__(*args, position=position, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power [W]"

    def where_is_growing(self,
                         minimum_number_of_points: int = 50,
                         n_trailing_points_to_check: int = 40,
                         **kwargs) -> list[bool]:
        """Determine where power is growing (``True``) and where it is not.

        .. todo::
            May be necessary to also remove isolated True

        """
        n_points = self._raw_data.index[-1]
        is_growing: list[bool] = []

        previous_value = True
        for i in range(n_points):
            local_is_growing = array_is_growing(
                self.data,
                i,
                undetermined_value=previous_value,
                **kwargs)

            is_growing.append(local_is_growing)
            previous_value = local_is_growing

        arr_growing = np.array(is_growing, dtype=np.bool_)

        # Remove isolated False
        if minimum_number_of_points > 0:
            arr_growing = remove_isolated_false(arr_growing,
                                                minimum_number_of_points)

        # Also ensure that last power growth is False
        if n_trailing_points_to_check > 0:
            arr_growing = remove_trailing_true(
                arr_growing,
                n_trailing_points_to_check,
                array_name_for_warning='power growth')

        is_growing = arr_growing.tolist()
        return is_growing


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
