"""Define Penning to measure evolution of pressure."""

from multipac_testbench.instruments.instrument import Instrument


class Penning(Instrument):
    """A probe to measure pressure."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "Pressure [mbar]"
