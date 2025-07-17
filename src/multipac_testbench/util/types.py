"""Define specific and reusable types."""

from typing import Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Function/method to call when a post-treater is added to an
#: :class:`.Instrument`.
#:
#: .. seealso::
#:    :meth:`.Instrument.register_callback`
#:
CALLBACK_T = Callable[[], pd.Series]

#: Electric field probes names
FIELD_PROBES = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
FIELD_PROBES_T = Literal["E1", "E2", "E3", "E4", "E5", "E6", "E7"]

#: A function that takes in an :class:`.Instrument` data and return a boolean
#: array with same shape, indicating whether multipactor appeared
MULTIPAC_DETECTOR_T = Callable[[NDArray[np.float64]], NDArray[np.bool]]

#: A function that takes in the data of an instrument and returns an array with
#: same shape
POST_TREATER_T = Callable[[NDArray[np.float64]], NDArray[np.float64]]
