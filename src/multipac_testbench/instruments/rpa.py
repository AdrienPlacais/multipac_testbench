"""Define the RPA."""

import logging
from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from numpy.typing import NDArray


class RPAPotential(Instrument):
    """A probe to measure potential on RPA grid."""

    def __init__(self, *args, position: float = np.nan, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, position=position, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Grid potential [kV]"


class RPACurrent(Instrument):
    """A probe to measure collected current on RPA."""

    def __init__(
        self,
        *args,
        caliber_mA: float | None = None,
        position: float = np.nan,
        **kwargs,
    ) -> None:
        """Instantiate with the caliber.

        .. note::
            The current is automatically re-scaled to ``caliber_mA`` when this
            object is instantiated.

        Parameters
        ----------
        caliber_mA :
            Caliber in :unit:`mA`.

        """
        if caliber_mA is None:
            caliber_mA = 20.0
            logging.error(
                "The RPA current caliber was not given. Falling back on "
                f"default {caliber_mA =}."
            )
        self._caliber_mA = caliber_mA
        super().__init__(*args, position=position, **kwargs)
        self._recalibrate_current()

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"RPA current [$\mu$A]"

    def _recalibrate_current(self) -> None:
        r"""Rescale the measured data using the caliber.

        .. math::

            i_{real\,in\,mA} = i_{LabVIEW} * ``caliber_mA`` / 2

        """
        logging.debug(f"Rescaling RPA current with {self._caliber_mA = }")
        self._raw_data *= self._caliber_mA * 0.5


class RPA(VirtualInstrument):
    """Store the multipactor electrons energy distribution.

    This object is created by :meth:`.InstrumentFactory.run_virtual` when there
    is one :class:`.RPACurrent` and one :class:`.RPAPotential` in its
    ``instruments`` argument.

    """

    @classmethod
    def from_current_and_potential(
        cls,
        rpa_current: RPACurrent,
        rpa_potential: RPAPotential,
        name: str = "RPA",
        **kwargs,
    ) -> Self:
        """Compute the distribution from the current and grid potential."""
        averaged_current, corresponding_potentials = (
            _average_points_with_same_grid_potential(
                rpa_current.data, rpa_potential.data
            )
        )
        distribution = _compute_energy_distribution(
            averaged_current,
            corresponding_potentials,
        )
        ser_distribution = pd.Series(distribution, name=name)
        return cls(
            name=name, raw_data=ser_distribution, position=np.nan, **kwargs
        )

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Energy distribution [$\mu$A/kV]"


def _average_points_with_same_grid_potential(
    current: NDArray[np.float64], potential: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average the current.

    .. todo::
        Sometimes, you measure the RPA current several times with the same grid
        potential to improve accuracy. This method will average this RPA
        currents.
        Note: I will have to make a difference between two consecutive
        points, and points with the same grid potential but corresponding
        to increasing and decreasing power cycles.

    """
    logging.debug("Averaging RPA currents on same grid potential.")
    logging.error("RPA averaging not implemented yet.")
    return current, potential


def _compute_energy_distribution(
    averaged_current: NDArray[np.float64],
    corresponding_potentials: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Derivate signal to obtain distribution."""
    return np.diff(averaged_current) / np.diff(corresponding_potentials)
