"""Define voltage probe to measure potential on RPA grid."""

from multipac_testbench.instruments.instrument import Instrument


class RPAPotential(Instrument):
    """A probe to measure potential on RPA grid."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Grid potential [V]"
