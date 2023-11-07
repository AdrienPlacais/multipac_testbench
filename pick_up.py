#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to keep measurements at a pick-up."""
from dataclasses import dataclass

import numpy as np
from matplotlib.axes._axes import Axes


@dataclass
class PickUp:
    """Hold information on a single pick-up."""

    name: str
    position: float
    _sample_index: np.ndarray
    electric_field_probe: np.ndarray
    mp_current_probe: np.ndarray
    n_mp_zones: int | None = None

    def __post_init__(self):
        """Compute multipactor limits."""
        if True:
            return
        if self.n_mp_zones is None:
            n_mp_zones = self._determine_number_of_multipactor_zones()

        mp_limits_idx = {i: self._get_multipactor_indexes(i)
                         for i in range(n_mp_zones)}
        mp_voltage = {i: self._get_multipactor_voltage(mp_limits_idx[i])
                      for i in range(n_mp_zones)}
        self.mp_voltage = mp_voltage

    def __str__(self) -> str:
        """Print the name of the pick up as well as the voltage limits."""
        return f"Probe {self.name}, mp zones: {self.mp_voltage}"

    def _determine_number_of_multipactor_zones(self) -> int:
        """Determine how many multipactor zones there is if not provided."""
        raise NotImplementedError("Please specify manually the number of "
                                  "mp zones to expect.")

    # handle mp that does not end
    def _get_multipactor_indexes(self) -> tuple[int, int]:
        """Compute index of multipactor start and end."""
        return 0, 0

    # handle mp that does not end
    def _get_multipactor_voltage(self, mp_limits_idx: tuple[int, int]
                                 ) -> tuple[float, float]:
        """Compute multipactor voltage start and end."""
        start = self.electric_field_probe[mp_limits_idx[0]]
        end = self.electric_field_probe[mp_limits_idx[1]]
        return (start, end)

    def plot_electric_field(self, axx: Axes) -> None:
        """Plot the electric field as a function of sample index."""
        axx.plot(self._sample_index, self.electric_field_probe,
                 label=self.name)

    def plot_mp_current(self, axx: Axes) -> None:
        """Plot the electron pick-up current vs sample index."""
        axx.plot(self._sample_index, self.mp_current_probe, label=self.name)
