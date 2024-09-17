"""Define current probe to measure multipactor cloud current."""

from multipac_testbench.instruments.instrument import Instrument


class CurrentProbe(Instrument):
    """A probe to measure multipacting current."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Multipactor current [$\mu$A]"
