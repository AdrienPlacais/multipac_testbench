"""Define current probe associated with RPA."""

from multipac_testbench.instruments.instrument import Instrument


class RPACurrent(Instrument):
    """A probe to measure collected current on RPA."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"RPA current [$\mu$A]"
