"""Define the RPA."""

import logging
from functools import partial
from typing import Any, Self, cast

import numpy as np
import pandas as pd
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from multipac_testbench.util.helper import drop_repeated_col
from multipac_testbench.util.post_treaters import (
    average_y_for_nearby_x_within_distance,
    drop_x_where_y_is_nan,
)


class RPAPotential(Instrument):
    """A probe to measure potential on RPA grid."""

    def __init__(self, *args, position: float = np.nan, **kwargs) -> None:
        """Instantiate object and convert signal to :unit:`V`."""
        super().__init__(*args, position=position, **kwargs)
        self._raw_data *= 1e3

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Grid potential [V]"


class RPACurrent(Instrument):
    """A probe to measure collected current on RPA."""

    def __init__(
        self,
        *args,
        caliber_mA: float | None = None,
        position: float = np.nan,
        average: bool = False,
        max_index_distance: int | None = None,
        tol: float | None = None,
        keep_shape: bool | None = None,
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
        average :
            If current should be averaged at nearly identical potentials. If
            set to ``True``, averaging will be performed with ``tol`` and
            ``max_index_distance`` ``kwargs`` using
            :func:`.average_y_for_nearby_x_within_distance`.
        tol :
            Maximum absolute difference under which potential values are
            considered equal.
        max_index_distance :
            Maximum index separation allowed when grouping similar potentials.
            Prevents averaging across distant, unrelated measurements.
        keep_shape :
            If ``True``, the returned array has the same shape as the input,
            with only the first element of each group containing the average
            and others filled with ``np.nan``. If ``False``, returns a compact
            array with only the averaged values.

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

        self.avg_kwargs: dict[str, Any]
        if not average:
            return
        self.avg_kwargs = self._averaging_kwargs(
            max_index_distance=max_index_distance,
            tol=tol,
            keep_shape=keep_shape,
        )

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

    def _averaging_kwargs(
        self,
        max_index_distance: int | None = None,
        tol: float | None = None,
        keep_shape: bool | None = None,
    ) -> dict[str, Any]:
        """Set the ``kwargs`` for the averaging function.

        The post-treater function performing the average is created at the
        instantiation of :class:`RPA`.

        Parameters
        ----------
        tol :
            Maximum absolute difference under which potential values are
            considered equal.
        max_index_distance :
            Maximum index separation allowed when grouping similar potentials.
            Prevents averaging across distant, unrelated measurements.
        keep_shape :
            If ``True``, the returned array has the same shape as the input,
            with only the first element of each group containing the average
            and others filled with ``np.nan``. If ``False``, returns a compact
            array with only the averaged values.

        """
        avg_kwargs = {}
        if max_index_distance is not None:
            avg_kwargs["max_index_distance"] = max_index_distance
        if tol is not None:
            avg_kwargs["tol"] = tol
        if keep_shape is not None:
            avg_kwargs["keep_shape"] = keep_shape
        return avg_kwargs


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
        if hasattr(rpa_current, "avg_kwargs"):
            _set_up_current_averaging(rpa_current, rpa_potential)

        distribution = _compute_energy_distribution(
            rpa_potential.data_as_pd, rpa_current.data_as_pd
        )
        return cls(
            name=name,
            raw_data=distribution,
            position=np.nan,
            is_2d=True,
            **kwargs,
        )

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Energy distribution [$\mu$A/V]"


def _set_up_current_averaging(
    rpa_current: RPACurrent, rpa_potential: RPAPotential
) -> None:
    """Average RPA current at nearly identical RPA potentials.

    Also remove duplicate ``rpa_potential`` data to enforce both instruments to
    hae same shape.

    """
    averager = partial(
        average_y_for_nearby_x_within_distance,
        x_values=rpa_potential.data,
        **rpa_current.avg_kwargs,
    )
    rpa_current.add_post_treater(averager)

    keep_shape = averager.keywords.get("keep_shape", True)
    if keep_shape:
        return

    logging.warning(
        "The RPA current averager will alter the shape of data, which may "
        "cause issues. I will try to adapt, but you may have to go back to "
        "keep_shape = False."
    )
    shape_consistency_enforcer = partial(
        drop_x_where_y_is_nan, y_values=rpa_current.data
    )
    rpa_potential.add_post_treater(shape_consistency_enforcer)
    return


def _compute_energy_distribution(
    potential: pd.Series | pd.DataFrame, current: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """Derive signal to obtain distribution."""
    df = pd.concat([potential, current], axis=1)
    assert isinstance(df, pd.DataFrame)
    df = drop_repeated_col(df)

    dropped_potential = df.iloc[:, 0]
    dropped_current = df.iloc[:, 1]

    distribution = -dropped_current.diff() / dropped_potential.diff()
    distribution.name = "Energy distribution"

    out = pd.concat([dropped_potential, distribution], axis=1)
    return out
