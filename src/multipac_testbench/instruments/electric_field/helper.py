"""Define utility functions for electric fields."""

from pathlib import Path

import numpy as np
import pandas as pd
from multipac_testbench.util.types import FIELD_PROBES


def read_e_field_probe_calibration(data: pd.DataFrame) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Read the field probe attenuation, rf racks calibration from data file.

    Parameters
    ----------
    data :
        Data as returned by the :func:`.loader.load` function.

    Returns
    -------
    g_probe : dict[str, float]
        Associates every field probe name (eg ``"E1"``) with it's attenuation
        in :unit:`dBm`.
    a_rack : dict[str, float]
        Associates every field probe name (eg ``"E1"``) with it's rack
        calibration slope in :unit:`dBm/V`.
    b_rack : dict[str, float]
        Associates every field probe name (eg ``"E1"``) with it's rack
        calibration offset in :unit:`dBm`.

    Raises
    ------
    NotImplementedError

    """
    raise NotImplementedError


def load_rf_calibration_files(
    calibration_folder: Path,
    freq_mhz: float,
    freq_col: str = "Frequency [MHz]",
    a_col: str = "a [dBm / V]",
    b_col: str = "b [dBm]",
) -> tuple[dict[str, float], dict[str, float]]:
    """Load rf calibration files, interpolate proper calibration data.

    ``calibration_folder`` must look like:

        .. code-block::

            calibration_folder/
            ├── E1_fit_calibration.csv
            ├── E2_fit_calibration.csv
            ├── E3_fit_calibration.csv
            ├── E4_fit_calibration.csv
            ├── E5_fit_calibration.csv
            ├── E6_fit_calibration.csv
            └── E7_fit_calibration.csv

    Where the given ``CSV`` must hold data like:

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
    calibration_folder :
        Path to the folder holding ``CSV`` calibration files.
    freq_mhz :
        RF frequency for this test in :unit:`MHz`.
    freq_col :
        Name of the column holding the measure frequency in :unit:`MHz`.
    a_col :
        Name of the column holding the measured slope in :unit:`dBm/V`.
    b_col :
        Name of the column holding the measured bias in :unit:`dBm`.

    """
    data = {
        probe: pd.read_csv(
            calibration_folder / f"{probe}_fit_calibration.csv",
            sep="\t",
            comment="#",
            index_col=freq_col,
            usecols=[a_col, b_col, freq_col],
        )
        for probe in FIELD_PROBES
    }

    a_rack: dict[str, float] = {}
    b_rack: dict[str, float] = {}

    for probe, df in data.items():
        if freq_mhz not in df.index:
            df.loc[freq_mhz] = [np.nan, np.nan]
            df.sort_index(inplace=True)
            df.interpolate(inplace=True)
        ser = df.loc[freq_mhz]

        a_rack[probe] = float(ser[a_col])
        b_rack[probe] = float(ser[b_col])
    return a_rack, b_rack
