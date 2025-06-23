"""Define various instruments transfer functions."""

import numpy as np
from numpy.typing import NDArray


def v_coax_to_v_acquisition(
    v_coax: NDArray[np.float64],
    g_probe: float,
    a_rack: float,
    b_rack: float,
    z_0: float = 50.0,
) -> NDArray[np.float64]:
    r"""Convert coaxial voltage to acquisition voltage.

    This is the inverse of the function that is implemented in LabVIEWER.

    Parameters
    ----------
    v_coax :
        :math:`V_\mathrm{coax}` in :unit:`V`, which should be the content of
        the ``NI9205_Ex`` columns.
    g_probe :
        Total attenuation. Probe specific, also depends on frequency.
    a_rack :
        Rack calibration slope in :unit:`dBm/V`.
    b_rack :
        Rack calibration constant in :unit:`dBm`.
    z_0 :
        Line impedance in :unit:`\\Omega`.

    Returns
    -------
    v_acq :
        Acquisition voltage in :math:`[0, 10~\mathrm{V}]`.

    """
    p_w = v_coax**2 / (2.0 * z_0)
    p_dbm = 30.0 + 10.0 * np.log10(p_w)
    p_acq = p_dbm - abs(g_probe + 3.0)
    v_acq = ((p_acq - b_rack) / a_rack).astype(np.float64)
    return v_acq


def v_acquisition_to_v_coax(
    v_acq: NDArray[np.float64],
    g_probe: float,
    a_rack: float,
    b_rack: float,
    z_0: float = 50.0,
) -> NDArray[np.float64]:
    r"""Convert acquisition voltage to coaxial voltage.

    This is the same function that is implemented in LabVIEWER.

    Parameters
    ----------
    v_acq :
        Acquisition voltage in :math:`[0, 10~\mathrm{V}]`.
    g_probe :
        Total attenuation. Probe specific, also depends on frequency.
    a_rack :
        Rack calibration slope in :unit:`dBm/V`.
    b_rack :
        Rack calibration constant in :unit:`dBm`.
    z_0 :
        Line impedance in :unit:`\\Omega`.

    Returns
    -------
    v_coax :
        :math:`V_\mathrm{coax}` in :unit:`V`, which should be the content of
        the ``NI9205_Ex`` columns.

    """
    p_acq = v_acq * a_rack + b_rack
    p_dbm = abs(g_probe + 3.0) + p_acq
    p_w = 10 ** ((p_dbm - 30.0) / 10.0)
    v_coax = np.sqrt(2.0 * z_0 * p_w)
    return v_coax
