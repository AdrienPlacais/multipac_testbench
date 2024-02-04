#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define functions to create susceptiblity plots."""
import numpy as np


def measured_to_susceptibility_coordinates(
        voltages_v: np.ndarray | list[float],
        d_cm: float,
        freq_mhz: float,
        ) -> np.ndarray:
    """Convert measured data to coordinates for susceptibility plot."""
    y_coordinates = voltages_v
    n_coordinates = len(y_coordinates)
    x_coordinates = np.full(
        n_coordinates,
        d_cm * freq_mhz,
    )
    coordinates = np.column_stack((x_coordinates, y_coordinates))
    return coordinates
