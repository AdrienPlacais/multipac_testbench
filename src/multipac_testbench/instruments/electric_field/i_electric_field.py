"""Define mother class for all instruments measuring electric fields."""

import pandas as pd
from multipac_testbench.instruments.instrument import Instrument


class IElectricField(Instrument):
    """A generic instrument for electric fields."""

    def __init__(
        self,
        name: str,
        raw_data: pd.Series | None,
        **kwargs,
    ) -> None:
        """Instantiate the class."""
        super().__init__(name, raw_data, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Voltage [V]"
