"""Define field probe to measure electric field."""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from multipac_testbench.instruments.electric_field.i_electric_field import (
    IElectricField,
)
from multipac_testbench.util.post_treaters import (
    v_acquisition_to_v_coax,
    v_coax_to_v_acquisition,
)


class FieldProbe(IElectricField):
    """A probe to measure electric field."""

    def __init__(
        self,
        *args,
        g_probe: float | None = None,
        calibration_file: str | None = None,
        patch: bool = False,
        **kwargs,
    ) -> None:
        r"""Instantiate with some specific arguments.

        Parameters
        ----------
        g_probe :
            Total attenuation. Probe specific, also depends on frequency.
        a_rack :
            Rack calibration slope in :unit:`V/dBm`.
        b_rack :
            Rack calibration constant in :unit:`dBm`.

        """
        super().__init__(*args, **kwargs)
        self._g_probe = g_probe

        self._a_rack: float
        self._b_rack: float
        if calibration_file is not None:
            self._a_rack, self._b_rack = self._load_calibration_file(
                Path(calibration_file)
            )
        if patch:
            self._patch_data()

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Measured voltage [V]"

    def _patch_data(self, g_probe_in_labview: float = -1.0) -> None:
        """Correct  ``raw_data`` when ``g_probe`` in LabVIEWER is wrong.

        The default value for ``g_probe_in_labview`` is only a guess.

        """
        assert hasattr(self, "_a_rack")
        assert hasattr(self, "_b_rack")
        assert self._g_probe is not None
        fun1 = partial(
            v_coax_to_v_acquisition,
            g_probe=g_probe_in_labview,
            a_rack=self._a_rack,
            b_rack=self._b_rack,
            z_0=50.0,
        )
        fun2 = partial(
            v_acquisition_to_v_coax,
            g_probe=self._g_probe,
            a_rack=self._a_rack,
            b_rack=self._b_rack,
            z_0=50.0,
        )
        self._raw_data = fun1(self._raw_data)
        self._raw_data = fun2(self._raw_data)

    def _load_calibration_file(
        self,
        calibration_file: Path,
        freq_mhz: float = 120.0,
        freq_col: str = "Frequency [MHz]",
        a_col: str = "a [dBm / V]",
        b_col: str = "b [dBm]",
    ) -> tuple[float, float]:
        """Load calibration file, interpolate proper calibration data."""
        data = pd.read_csv(
            calibration_file,
            sep="\t",
            comment="#",
            index_col=freq_col,
            usecols=[a_col, b_col, freq_col],
        )
        if freq_mhz not in data.index:
            data.loc[freq_mhz] = [np.nan, np.nan]
            data.sort_index(inplace=True)
            data.interpolate(inplace=True)
        ser = data.loc[freq_mhz]
        a_rack = ser[a_col]
        b_rack = ser[b_col]
        return a_rack, b_rack
