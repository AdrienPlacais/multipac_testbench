#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to create the proper :class:`.IMeasurementPoint`."""
import pandas as pd

from multipac_testbench.src.instruments.factory import InstrumentFactory
from multipac_testbench.src.measurement_point.global_diagnostics import \
    GlobalDiagnostics
from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint
from multipac_testbench.src.measurement_point.pick_up import PickUp


class IMeasurementPointFactory:
    """Class to create the proper :class:`.GlobalDiagnostics` :class:`.PickUp`.

    It infers the proper type, position of instruments as well as the measured
    data from the configuration ``.toml`` file and the measurements ``.csv``
    file.

    """

    def __init__(self) -> None:
        """Instantiate the class with its :class:`InstrumentFactory`."""
        self.instrument_factory = InstrumentFactory()

    def run_single(self,
                   config_key: str,
                   config_value: dict,
                   df_data: pd.DataFrame
                   ) -> IMeasurementPoint:
        """Create a single measurement point.

        Parameters
        ----------
        config_key : str
            A key from the ``.toml`` file. If 'global' keyword is in the key,
            we return a :class:`.GlobalDiagnostics`. Else, we return a
            :class:`.PickUp`.
        config_value : dict
            Values from the ``.toml`` file corresponding to ``config_key``,
            which will passed down to the created :class:`.IMeasurementPoint`.
        df_data : pd.DataFrame
            Full data from the ``.csv`` file.

        Returns
        -------
        IMeasurementPoint
            A :class:`.GlobalDiagnostics` or :class:`.PickUp`.

        """
        if "global" in config_key:
            return GlobalDiagnostics(
                name=config_key,
                df_data=df_data,
                instrument_factory=self.instrument_factory,
                **config_value)
        return PickUp(name=config_key,
                      df_data=df_data,
                      instrument_factory=self.instrument_factory,
                      **config_value)

    def run(self,
            config: dict[str, dict],
            df_data: pd.DataFrame
            ) -> tuple[GlobalDiagnostics | None, list[PickUp]]:
        """Create all the measurement points."""
        measurement_points = [
            self.run_single(config_key, config_value, df_data)
            for config_key, config_value in config.items()
        ]

        global_diagnostics = self._filter_global_diagnostics(
            measurement_points)
        pick_ups = self._filter_pick_ups(measurement_points)
        return global_diagnostics, pick_ups

    def _filter_global_diagnostics(self,
                                   measurement_points: list[IMeasurementPoint],
                                   ) -> GlobalDiagnostics | None:
        """Ensure that we have only one :class:GlobalDiagnostics` object."""
        global_diagnostics = [x for x in measurement_points
                              if isinstance(x, GlobalDiagnostics)
                              ]
        if len(global_diagnostics) == 0:
            print("No global diagnostic defined.")
            return
        if len(global_diagnostics) == 1:
            print("1 set of global diagnostics defined:\n\t"
                  f"{global_diagnostics[0]}")
            return global_diagnostics[0]

        raise IOError("Several global diagnostics were found! It means that"
                      " several entries in the .toml file have the word "
                      "'global' in their entries. Please gather them.")

    def _filter_pick_ups(self,
                         measurement_points: list[IMeasurementPoint]
                         ) -> list[PickUp]:
        """Print information on the created pick-ups."""
        pick_ups = [x for x in measurement_points if isinstance(x, PickUp)]
        n_pick_ups = len(pick_ups)
        if len(pick_ups) == 0:
            raise IOError("No pick-up was defined.")

        print(f"{n_pick_ups} pick-ups created:")
        for pick_up in pick_ups:
            print(f"\t{pick_up}")

        return pick_ups