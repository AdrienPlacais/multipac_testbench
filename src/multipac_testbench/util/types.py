"""Define specific and reusable types."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

#: A function that takes in the data of an instrument and returns an array with
#: same shape
POST_TREATER_T = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: A function that takes in an :class:`.Instrument` data and return a boolean
#: array with same shape, indicating whether multipactor appeared
MULTIPAC_DETECTOR_T = Callable[[NDArray[np.float64]], NDArray[np.bool]]
