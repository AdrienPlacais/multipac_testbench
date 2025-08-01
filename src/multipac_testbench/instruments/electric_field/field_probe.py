"""Define field probe to measure electric field."""

import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from multipac_testbench.instruments.electric_field.i_electric_field import (
    IElectricField,
)
from multipac_testbench.util.files import resolve_path
from multipac_testbench.util.transfer_functions import field_probe
from multipac_testbench.util.types import POST_TREATER_T


class FieldProbe(IElectricField):
    """A probe to measure electric field."""

    def __init__(
        self,
        name: str,
        raw_data: pd.Series | None,
        attenuation_file: str | None = None,
        calibration_file: str | None = None,
        freq_mhz: float | None = None,
        **kwargs,
    ) -> None:
        r"""Instantiate with some specific arguments.

        See Also
        --------
        :func:`.transfer_functions.field_probe`

        Parameters
        ----------
        attenuation_file :
            Path to the probe attenuation file, linking voltage in line to
            voltage measured by the probe. This is frequency and probe
            specific.
        calibration_file :
            Path to the probe calibration file, linking the probe voltage (sent
            to the National Instruments card) to the actual voltage in the tube
            at the probe position. Check :meth:`_rf_rack_calibration_constants`
            for more information. Used when the ``g_probe`` in LabViewer is
            wrong and data must be patched (``patch == True``, or when
            ``raw_data``).

        """
        #: Total attenuation. Probe specific, also depends on frequency.
        self._g_probe: float
        #: Rack calibration slope in :unit:`V/dBm`.
        self._a_rack: float
        #: Rack calibration constant in :unit:`dBm`.
        self._b_rack: float

        if calibration_file is not None:
            assert (
                freq_mhz is not None
            ), "Frequency is mandatory to calibrate racks."
            self._a_rack, self._b_rack = self._rf_rack_calibration_constants(
                calibration_file,
                freq_mhz=freq_mhz,
            )
        if attenuation_file is not None:
            assert (
                freq_mhz is not None
            ), "Frequency is mandatory to calibrate probes."
            self._g_probe = self._probe_attenuation(
                Path(attenuation_file), freq_mhz=freq_mhz, name=name
            )
        self._files = {
            "attenuation": attenuation_file,
            "calibration": calibration_file,
        }
        super().__init__(name=name, raw_data=raw_data, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Measured voltage [V]"

    @property
    def _transfer_functions(self) -> list[POST_TREATER_T]:
        assert hasattr(self, "_a_rack")
        assert hasattr(self, "_b_rack")
        assert hasattr(self, "_g_probe")
        return [
            partial(
                field_probe,
                g_probe=self._g_probe,
                a_rack=self._a_rack,
                b_rack=self._b_rack,
                z_0=50.0,
            )
        ]

    def _rf_rack_calibration_constants(
        self,
        calibration_file: Path | str,
        freq_mhz: float,
        freq_col: str = "Frequency [MHz]",
        a_col: str = "a [dBm / V]",
        b_col: str = "b [dBm]",
    ) -> tuple[float, float]:
        """Load calibration file, interpolate proper calibration data.

        .. todo::
            To refactor so that it takes ``calibration_folder`` as argument
            instead of ``calibration_file``. Idea is to be more DRY.

        The given file must look like:

        .. code-block::

            # some comments
            Probe	Frequency [MHz]	a [dBm / V]	b [dBm]
            E1	80.0	10.232945073011583	-51.43251555580861
            E1	88.0	10.244590821913084	-51.46188696517617
            E1	100.0	10.270347916270323	-51.578312368686596
            E1	120.0	10.301710211286146	-51.73648053093371
            E1	140.0	10.33558455881163	-51.83334288966003
            E1	160.0	10.375268145607556	-51.91758233328844
            E1	180.0	10.398407751401276	-51.87058673739318

        The preferred way to create such a file is to use `the dedicated
        tool`_.

        .. _`the dedicated tool`: https://github.com/AdrienPlacais/multipac_testbench_calibrate_racks

        Parameters
        ----------
        calibration_file :
            Path to the ``CSV`` calibration file.
        freq_mhz :
            RF frequency for this test in :unit:`MHz`.
        freq_col :
            Name of the column holding the measure frequency in :unit:`MHz`.
        a_col :
            Name of the column holding the measured slope in :unit:`dBm/V`.
        b_col :
            Name of the column holding the measured bias in :unit:`dBm`.

        """
        data = pd.read_csv(
            resolve_path(calibration_file),
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

    def _probe_attenuation(
        self,
        attenuation_file: Path | str,
        freq_mhz: float,
        name: str,
    ) -> float:
        """Load attenuation file, interpolate proper attenuation data.

        The given file must look like:

        .. code-block::

            # Calibration of electric field probes
            # Adrien Placais measurement on 2025-06-11
            # Attenuations are in dB
            Frequency [MHz],100,120,140,160
            NI9205_E1,-78.7,-77.2,-75.6,-75.4
            NI9205_E2,-77.8,-77.4,-77.2,-75.4
            NI9205_E3,-78.1,-77.2,-76.8,-76.6
            NI9205_E4,-77.8,-76.8,-75.9,-74.6
            NI9205_E5,-79.5,-76.9,-76.4,-75.5
            NI9205_E6,-79.6,-78.2,-77.5,-76.9
            NI9205_E7,-75.9,-76.6,-74.4,-74.0

        Parameters
        ----------
        attenuation_file :
            Path to the ``CSV`` attenuation file.
        freq_mhz :
            RF frequency for this test in :unit:`MHz`. If not present in the
            file, it is interpolated. If it is outside interpolation range, a
            warning is printed.
        name :
            Name of current column; must correspond to a line in the file.

        Returns
        -------
        g_probe : float
            Attenuation for this probe at ``freq_mhz``.

        """
        df = pd.read_csv(
            resolve_path(attenuation_file), comment="#", index_col=0
        )

        df.columns = df.columns.astype(float)

        if name not in df.index:
            raise ValueError(f"Probe '{name}' not found in attenuation file")

        freqs = df.columns.to_numpy()
        attens = df.loc[name].to_numpy()

        if freq_mhz < freqs[0] or freq_mhz > freqs[-1]:
            logging.warning(
                f"Frequency {freq_mhz} MHz is outside the calibration range "
                f"({freqs[0]}--{freqs[-1]} MHz). Extrapolating."
            )

        return float(np.interp(freq_mhz, freqs, attens))
