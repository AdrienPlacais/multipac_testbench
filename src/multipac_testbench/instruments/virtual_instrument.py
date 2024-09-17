"""Define an instrument-like object."""

import pandas as pd
from multipac_testbench.instruments.instrument import Instrument


class VirtualInstrument(Instrument):
    """An object that works like an :class:`.Instrument`.

    Allows to avoid confusion when the object under study should have same
    methods than a classic instrument, but is user-defined with analytical data
    or data calculated from other instruments.

    """

    def __init__(self, name: str, raw_data: pd.Series, **kwargs) -> None:
        """Instantiate object."""
        super().__init__(name, raw_data, **kwargs)
