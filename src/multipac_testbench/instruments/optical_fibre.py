"""Define optical fivre to detect multipactor arcs.

.. todo::
    Find out the units for optical fibre.

"""

from multipac_testbench.instruments.instrument import Instrument


class OpticalFibre(Instrument):
    """An optical fibre detecting multipacting light."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Optical fibre signal [unit?]"
