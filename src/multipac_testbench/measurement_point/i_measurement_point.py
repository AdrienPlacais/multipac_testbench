"""Define an object to keep several related measurements."""

import logging
from abc import ABC, ABCMeta
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from multipac_testbench.instruments.factory import InstrumentFactory
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.multipactor_band.instrument_multipactor_bands import (
    InstrumentMultipactorBands,
)
from multipac_testbench.util.types import MULTIPAC_DETECTOR_T, POST_TREATER_T
from numpy.typing import NDArray


class IMeasurementPoint(ABC):
    """Hold several related measurements.

    In particular, gather :class:`.Instrument` which have the same position.

    """

    def __init__(
        self,
        name: str,
        df_data: pd.DataFrame,
        instrument_factory: InstrumentFactory,
        instruments_kw: dict[str, dict[str, Any]],
        position: float,
        color: str | None = None,
    ) -> None:
        """Create the all the global instruments.

        Parameters
        ----------
        df_data :
            df_data
        instrument_factory :
            An object that creates :class:`.Instrument`.
        instruments_kw :
            Dictionary which keys are name of the column where the data from
            the instrument is. Values are dictionaries with keyword arguments
            passed to the proper :class:`.Instrument`.
        position :
            Position of the measurement point. It is a real if it is a
            :class:`.PickUp`, and ``np.nan`` for a :class:`.GlobalDiagnostics`.
        color :
            HTML color of the plots. It is ``None`` for
            :class:`.GlobalDiagnostics`.

        """
        self.name = name
        self.position = position
        self.color = color
        #: Holds all :class:`.Instrument` instances at this location
        self.instruments = [
            instrument_factory.run(
                instr_name, df_data, color=color, **instr_kw
            )
            for instr_name, instr_kw in instruments_kw.items()
        ]
        virtual_instruments = instrument_factory.run_virtual(
            self.instruments,
            is_global=np.isnan(position),
        )
        self.add_instrument(*virtual_instruments)

    def add_instrument(self, *instruments: Instrument) -> None:
        """Add a new :class:`.Instrument` to :attr:`instruments`.

        A priori, useful only for :class:`.VirtualInstrument`, when they rely
        on other :class:`.Instrument` objects to be fully initialized.

        """
        for instrument in instruments:
            self.instruments.append(instrument)

    def get_instruments(
        self,
        instrument_class: ABCMeta,
        instruments_to_ignore: Sequence[Instrument | str] = (),
    ) -> list[Instrument]:
        """Get instruments which are (sub) classes of ``instrument_class``.

        An empty list is returned when current pick-up has no instrument of the
        desired instrument class.

        """
        instrument_names_to_ignore = [
            x if isinstance(x, str) else x.name for x in instruments_to_ignore
        ]
        instruments = [
            instrument
            for instrument in self.instruments
            if isinstance(instrument, instrument_class)
            and instrument.name not in instrument_names_to_ignore
        ]
        return instruments

    def get_instrument(self, *args, **kwargs) -> Instrument | None:
        """Get instrument which is (sub) class of ``instrument_class``.

        Raise an error if several instruments match the condition.

        """
        instruments = self.get_instruments(*args, **kwargs)
        if len(instruments) == 0:
            return
        if len(instruments) == 1:
            return instruments[0]
        raise OSError(
            f"More than one instrument found with {args = } and {kwargs = }."
        )

    def add_post_treater(
        self,
        post_treater: POST_TREATER_T,
        instrument_class: ABCMeta = Instrument,
        verbose: bool = False,
    ) -> None:
        """Add post-treatment functions to instruments."""
        instruments = self.get_instruments(instrument_class)
        for instrument in instruments:
            instrument.add_post_treater(post_treater)

            if verbose:
                logging.info(f"A post-treater was added to {str(instrument)}.")

    def detect_multipactor(
        self,
        multipac_detector: MULTIPAC_DETECTOR_T,
        instrument_class: ABCMeta,
        growth_mask: NDArray[np.bool],
        debug: bool = False,
        info: str = "",
    ) -> InstrumentMultipactorBands | None:
        """Detect multipactor with ``multipac_detector``."""
        instrument = self.get_instrument(instrument_class)
        if instrument is None:
            return
        multipactor = multipac_detector(instrument.data)
        instrument_multipactor_bands = InstrumentMultipactorBands(
            multipactor,
            growth_mask,
            instrument.name,
            self.name,
            instrument.position,
            info,
            color=self.color,
        )
        if debug:
            axes = instrument.data_as_pd.plot(grid=True)
            axes = axes.twinx()
            df_float = pd.DataFrame(
                {
                    "Power grows?": growth_mask,
                    "Multipactor?": multipactor[1:],
                }
            ).astype(float)
            axes = df_float.plot(ax=axes, grid=True)

        return instrument_multipactor_bands

    def scatter_instruments_data(
        self,
        instrument_class_axes: dict[ABCMeta, Axes],
        xdata: float,
        instrument_multipactor_bands: InstrumentMultipactorBands,
    ) -> None:
        """Scatter data measured by desired instruments."""
        for instrument_class, axes in instrument_class_axes.items():
            instrument = self.get_instrument(instrument_class)
            if instrument is None:
                continue

            instrument.scatter_data(
                axes, instrument_multipactor_bands.multipactor, xdata
            )
