"""Define functions to fix measurement errors."""

import logging
from pathlib import Path

from multipac_testbench.instruments.electric_field.helper import (
    load_rf_calibration_files,
    read_e_field_probe_calibration,
)
from multipac_testbench.multipactor_test.loader import load, save
from multipac_testbench.util.transfer_functions import (
    field_probe,
    field_probe_inv,
)
from multipac_testbench.util.types import FIELD_PROBES


def fix_wrong_e_field_calibration(
    freq_mhz: float,
    g_probe_new: dict[str, float],
    rack_calibration_folder: Path,
    filepath_bad: Path,
    filepath_new: Path | None = None,
    g_probe_bad: dict[str, float] | None = None,
    a_rack_bad: dict[str, float] | None = None,
    b_rack_bad: dict[str, float] | None = None,
    sep: str = ",",
    **kwargs,
) -> None:
    """Recover proper electric field measurements.

    Use this when the attenuation and/or the rf rack data in LabView was
    incorrect.

    Parameters
    ----------
    freq_mhz :
        Test frequency in :unit:`MHz`.
    g_probe_new :
        Correct attenuation at current frequency for every field probe. Keys
        are the names of the field probes, *eg* ``"E1"``, ``"E2"``, etc.
    rack_calibration_folder :
        Folder holding all the rack calibration files. It should look like:

        .. code-block::

            outputs
            ├── E1_fit_calibration.csv
            ├── E2_fit_calibration.csv
            ├── E3_fit_calibration.csv
            ├── E4_fit_calibration.csv
            ├── E5_fit_calibration.csv
            ├── E6_fit_calibration.csv
            └── E7_fit_calibration.csv

    filepath_bad :
        Path to the original data file, generally ``XLSX`` file.
    filepath_new :
        Path to the new corrected data file, generally ``CSV`` file.
    g_probe_bad, a_rack_bad, b_rack_bad :
        Dictionaries linking every field probe to the electric field probe
        parameters as found in ``filepath_bad``. If not provided, should be
        read from ``filepath_bad`` directly.
    sep :
        Column delimiter in input and output ``CSV`` files.

    Raises
    ------
    NotImplementedError
        When the bad electric field probes parameters are not provided and
        they should be read from ``filepath_bad`` directly.

    """
    data = load(filepath_bad, sep=sep)

    if not all((g_probe_bad, a_rack_bad, b_rack_bad)):
        g_probe_bad, a_rack_bad, b_rack_bad = read_e_field_probe_calibration(
            data
        )
    assert g_probe_bad is not None
    assert a_rack_bad is not None
    assert b_rack_bad is not None

    a_rack_new, b_rack_new = load_rf_calibration_files(
        rack_calibration_folder, freq_mhz
    )

    for probe in FIELD_PROBES:
        col = f"NI9205_{probe}"
        if col not in data:
            continue

        v_acqui = field_probe_inv(
            data[col].to_numpy(),
            g_probe_bad[probe],
            a_rack_bad[probe],
            b_rack_bad[probe],
        )
        v_coax = field_probe(
            v_acqui, g_probe_new[probe], a_rack_new[probe], b_rack_new[probe]
        )
        data[col] = v_coax

    if filepath_new is None:
        logging.warning(
            "filepath_new was not given. I will overwrite the original data "
            "file. Is it ok for you? y/[n]"
        )
        answer = input()
        if answer not in ("y", "Y"):
            logging.info("Returning without overwritting.")
            return

        filepath_new = filepath_bad

    save(filepath_new, data, sep=sep, **kwargs)
