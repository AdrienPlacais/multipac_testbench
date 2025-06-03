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

        .. todo::
            Maybe ``a_rack``, ``b_rack`` and ``g_probe`` shoud be read from the
            results file.

        Parameters
        ----------
        g_probe :
            Total attenuation. Probe specific, also depends on frequency. Used
            when the ``g_probe`` in LabViewer is wrong and data must be
            patched (``patch == True``).
        calibration_file :
            Path to the probe calibration file, linking the probe voltage (sent
            to the National Instruments card) to the actual voltage in the tube
            at the probe position. Check :meth:`._load_calibration_file` for
            more information. Used when the ``g_probe`` in LabViewer is wrong
            and data must be patched (``patch == True``).
        patch :
            Patch the given data to compensate for a ``g_probe = -1`` in
            LabViewer. This was a bug that is now fixed, so this option should
            be useless now. See also :meth:`._patch_data`. The actual
            ``g_probe``, as well as the ``calibration_file`` argument.

        """
        super().__init__(*args, **kwargs)

        #: Total attenuation. Probe specific, also depends on frequency.
        self._g_probe = g_probe

        #: Rack calibration slope in :unit:`V/dBm`.
        self._a_rack: float
        #: Rack calibration constant in :unit:`dBm`.
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
        """Correct ``raw_data`` when ``g_probe`` in LabVIEWER is wrong.

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
        """Load calibration file, interpolate proper calibration data.

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
            euh
        freq_col :
            Name of the column holding the measure frequency in :unit:`MHz`.
        a_col :
            Name of the column holding the measured slope in :unit:`V/dBm`.
        b_col :
            Name of the column holding the measured bias in :unit:`dBm`.

        """
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
