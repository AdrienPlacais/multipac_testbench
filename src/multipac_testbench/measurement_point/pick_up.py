"""Define an object to keep measurements at a pick-up."""

from typing import Any

import numpy as np
import pandas as pd
from multipac_testbench.instruments.factory import InstrumentFactory
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)


class PickUp(IMeasurementPoint):
    """Hold measurements at a single pick-up."""

    def __init__(
        self,
        name: str,
        df_data: pd.DataFrame,
        instrument_factory: InstrumentFactory,
        instruments_kw: dict[str, dict[str, Any]],
        position: float,
        color: str,
    ) -> None:
        """Create the pick-up with all its instruments.

        Parameters
        ----------
        name :
            Name of the pick-up.
        df_data :
            df_data
        instrument_factory :
            An object that creates :class:`.Instrument`.
        position :
            Position pf the pick-up.
        instruments_kw :
            Dictionary which keys are name of the column where the data from
            the instrument is. Values are dictionaries with keyword arguments
            passed to the proper :class:`.Instrument`.
        color :
            HTML color that will be used for every instrument in this pick-up.

        """
        self._add_key_val_to_dictionaries("position", position, instruments_kw)
        assert not np.isnan(
            position
        ), "position = np.nan is reserved to global diagnostics."
        super().__init__(
            name,
            df_data,
            instrument_factory,
            instruments_kw,
            position=position,
            color=color,
        )

    def _add_key_val_to_dictionaries(
        self,
        key: str,
        value: Any,
        instruments_kw: dict[str, dict[str, Any]],
    ) -> None:
        """
        Add ``key``-``value`` pair to sub-dictionaries of ``instruments_kw``.

        In particular, used to instantiate every :class:`.Instrument` with its
        position.

        """
        for instr_kw in instruments_kw.values():
            instr_kw[key] = value

    def __str__(self) -> str:
        """Give concise info on pick-up."""
        out = f"""
        Pick-Up {self.name} at z = {self.position:1.3f}m,
        with instruments: {[str(x) for x in self.instruments]}
        """
        return " ".join(out.split())
