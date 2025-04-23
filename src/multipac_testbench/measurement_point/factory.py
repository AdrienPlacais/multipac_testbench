"""Define a class to create the proper :class:`.IMeasurementPoint`."""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from multipac_testbench.instruments.factory import InstrumentFactory
from multipac_testbench.measurement_point.global_diagnostics import (
    GlobalDiagnostics,
)
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from multipac_testbench.measurement_point.pick_up import PickUp


class IMeasurementPointFactory:
    """Class to create the proper :class:`.GlobalDiagnostics` :class:`.PickUp`.

    It infers the proper type, position of instruments as well as the measured
    data from the configuration ``TOML`` file and the measurements ``CSV``
    file.

    """

    def __init__(self, **kwargs) -> None:
        """Instantiate the class with its :class:`.InstrumentFactory`.

        Parameters
        ----------
        kwargs :
            Keyword arguments that are directly passed down to the
            :class:`.InstrumentFactory`.

        """
        self.instrument_factory = InstrumentFactory(**kwargs)

    def run_single(
        self,
        config_key: str,
        config_value: dict,
        df_data: pd.DataFrame,
        color: tuple[float, float, float],
    ) -> IMeasurementPoint:
        """Create a single measurement point.

        Parameters
        ----------
        config_key :
            A key from the ``TOML`` file. If 'global' keyword is in the key,
            we return a :class:`.GlobalDiagnostics`. Else, we return a
            :class:`.PickUp`.
        config_value :
            Values from the ``TOML`` file corresponding to ``config_key``,
            which will passed down to the created :class:`.IMeasurementPoint`.
        df_data :
            Full data from the ``CSV`` file.

        Returns
        -------
        measurement_point :
            A :class:`.GlobalDiagnostics` or :class:`.PickUp`.

        """
        if "global" in config_key:
            return GlobalDiagnostics(
                name=config_key,
                df_data=df_data,
                instrument_factory=self.instrument_factory,
                **config_value,
            )
        return PickUp(
            name=config_key,
            df_data=df_data,
            instrument_factory=self.instrument_factory,
            color=color,
            **config_value,
        )

    def run(
        self,
        config: dict[str, dict],
        df_data: pd.DataFrame,
        verbose: bool = False,
    ) -> tuple[GlobalDiagnostics | None, list[PickUp]]:
        """Create all the measurement points."""
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(colors)
        measurement_points = [
            self.run_single(
                config_key, config_value, df_data, colors[i % n_colors]
            )
            for i, (config_key, config_value) in enumerate(config.items())
        ]

        global_diagnostics = self._filter_global_diagnostics(
            measurement_points, verbose
        )
        pick_ups = self._filter_pick_ups(measurement_points, verbose)
        return global_diagnostics, pick_ups

    def _filter_global_diagnostics(
        self,
        measurement_points: list[IMeasurementPoint],
        verbose: bool = False,
    ) -> GlobalDiagnostics | None:
        """Ensure that we have only one :class:`.GlobalDiagnostics` object."""
        global_diagnostics = [
            x for x in measurement_points if isinstance(x, GlobalDiagnostics)
        ]
        if len(global_diagnostics) == 0:
            if verbose:
                logging.info("No global diagnostic defined.")
            return
        if len(global_diagnostics) == 1:
            if verbose:
                logging.info(
                    "1 set of global diagnostics defined:\n\t"
                    f"{global_diagnostics[0]}"
                )
            return global_diagnostics[0]

        raise OSError(
            "Several global diagnostics were found! It means that several "
            "entries in the ``TOML`` file have the word 'global' in their "
            "entry. Please gather them."
        )

    def _filter_pick_ups(
        self,
        measurement_points: list[IMeasurementPoint],
        verbose: bool = False,
    ) -> list[PickUp]:
        """Print information on the created pick-ups."""
        pick_ups = [x for x in measurement_points if isinstance(x, PickUp)]
        n_pick_ups = len(pick_ups)
        if len(pick_ups) == 0:
            raise OSError("No pick-up was defined.")

        if verbose:
            logging.info(f"{n_pick_ups} pick-ups created:")
            for pick_up in pick_ups:
                logging.info(f"\t{pick_up}")

        return pick_ups
