#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to keep measurements at a pick-up."""
from typing import Any

import numpy as np
import pandas as pd

from multipac_testbench.src.instruments.e_field_probe import ElectricFieldProbe
from multipac_testbench.src.instruments.factory import InstrumentFactory
from multipac_testbench.src.instruments.rf_power import RfPower
from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint


class PickUp(IMeasurementPoint):
    """Hold measurements at a single pick-up."""

    def __init__(self,
                 name: str,
                 df_data: pd.DataFrame,
                 instrument_factory: InstrumentFactory,
                 position: float,
                 instruments_kw: dict[str, dict[str, Any]],
                 ) -> None:
        """Create the pick-up with all its instruments.

        Parameters
        ----------
        name : str
            Name of the pick-up.
        df_data : pd.DataFrame
            df_data
        instrument_factory : InstrumentFactory
            An object that creates :class:`.Instrument`.
        position : float
            Position pf the pick-up.
        instruments_kw : dict[str, dict[str, Any]]
            Dictionary which keys are name of the column where the data from
            the instrument is. Values are dictionaries with keyword arguments
            passed to the proper :class:`.Instrument`.

        """
        self._add_key_val_to_dictionaries('position', position, instruments_kw)
        super().__init__(name,
                         df_data,
                         instrument_factory,
                         instruments_kw)
        self.position = position

    def _add_key_val_to_dictionaries(self,
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

    def create_rf_power_from_e_field_probe(self,
                                           swr: float | np.ndarray,
                                           impedance: float = 50.,
                                           name: str | None = None,
                                           **kwargs
                                           ) -> None:
        r"""Compute the power measured at this pick up from the e field probe.

        Parameters
        ----------
        swr : float | np.ndarray
            Voltage Signal Wave Ratio.
        impedance : float
            Impedance of the coaxial waveguide in :math:`\Ohm`.
        name : str | None
            Name given to the rf power probe.
        kwargs :
            Other keyword arguments.

        """
        e_field_probes = self.get_affected_instruments(ElectricFieldProbe)
        for e_field_probe in e_field_probes:
            rf_power_probe = RfPower.from_electric_field_probe(
                e_field_probe,
                swr,
                impedance=impedance,
                name=name,
                **kwargs,
                )
            self.instruments.append(rf_power_probe)
