"""Define instruments transfer functions.

These are the funtions converting acquisition voltages into actual physical
quantities.

"""

import numpy as np
from numpy.typing import NDArray


def current_probe(
    v_acq: NDArray[np.float64], a_probe: float
) -> NDArray[np.float64]:
    r"""Convert acquisition voltage to multipactor current.

    This is the transfer function of :class:`.CurrentProbe`.

    .. math::
        i = a_\mathrm{probe} \times V_\mathrm{acq}

    Parameters
    ----------
    v_acq :
        Acquisition voltage in :math:`[0, 10~\mathrm{V}]`.
    a :
        Calibration slope in :unit:`\\mu A/V`.

    Returns
    -------
        :math:`i` in :unit:`\\mu A`, which should be the content of the
        ``NI9205_MPxl`` columns.

    """
    return a_probe * v_acq


def field_probe(
    v_acq: NDArray[np.float64],
    g_probe: float,
    a_rack: float,
    b_rack: float,
    z_0: float = 50.0,
) -> NDArray[np.float64]:
    r"""Convert acquisition voltage to coaxial voltage.

    This is the transfer function of :class:`.FieldProbe`.

    .. math::
        P = \sqrt{2\times Z_0 \times 10^{-3} \times 10^{
            \frac{
                a_\mathrm{rack}V_\mathrm{acq} + b_\mathrm{rack}
                + G_\mathrm{probe} + 3\mathrm{dB}
            }{
                10
            }}}

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
        :math:`V_\mathrm{coax}` in :unit:`V`, which should be the content of
        the ``NI9205_Ex`` columns.

    """
    p_acq = v_acq * a_rack + b_rack
    p_dbm = abs(g_probe + 3.0) + p_acq
    p_w = 10 ** ((p_dbm - 30.0) / 10.0)
    v_coax = np.sqrt(2.0 * z_0 * p_w)
    return v_coax


def field_probe_inv(
    v_coax: NDArray[np.float64],
    g_probe: float,
    a_rack: float,
    b_rack: float,
    z_0: float = 50.0,
) -> NDArray[np.float64]:
    r"""Convert coaxial voltage to acquisition voltage.

    This is the inverse of the function that is implemented in LabViewer.

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
        Acquisition voltage in :math:`[0, 10~\mathrm{V}]`.

    """
    p_w = v_coax**2 / (2.0 * z_0)
    p_dbm = 30.0 + 10.0 * np.log10(p_w)
    p_acq = p_dbm - abs(g_probe + 3.0)
    v_acq = ((p_acq - b_rack) / a_rack).astype(np.float64)
    return v_acq


def power(
    v_acq: NDArray[np.float64],
    a_calib: float,
    b_calib: float,
    ensure_no_negative: bool = False,
) -> NDArray[np.float64]:
    r"""Convert acquisition voltage to power.

    This is the transfer function of :class:`.Power`.

    .. math::
        P = a_\mathrm{calib} \times V_\mathrm{acq} + b_\mathrm{calib}

    .. note::
        Original transfer function in LabView is:

        .. math::
            P = |V_\mathrm{acq}| \times (``REC_LIM_UPP`` - ``REC_LIM_LOW``)
            + ``REC_LIM_LOW``.

        We removed the absolute value; to avoid negative powers (should not
        appear), use ``ensure_no_negative=True``.

    Parameters
    ----------
    v_acq :
        Acquisition voltage in :math:`[0, 1~\mathrm{V}]` (not
        :math:`10~\mathrm{V}`!).
    a_calib :
        Calibration slope in :unit:`W/V`.
    b_calib :
        Calibration offset in :unit:`W`.
    ensure_no_negative :
        Set negative powers to :math:`0~\mathrm{V}`.

    Returns
    -------
        :math:`P` in :unit:`W`, which should be the content of the
        ``NI9205_Powerx`` columns.

    """
    watt = a_calib * v_acq + b_calib
    if ensure_no_negative:
        watt[watt < 0.0] = 0.0
    return watt


def power_channel_b(
    p_bad: NDArray[np.float64], k_fix: float, alpha_fix: float
) -> NDArray[np.float64]:
    r"""Fix power measured on channel B.

    Transfer function proposed by M. Vénière.

    .. math::
        P_\mathrm{ok} =
        k_\mathrm{fix} \times P_\mathrm{bad}^{\alpha_\mathrm{fix}}

    Parameters
    ----------
    p_bad :
        Power measured on channel B in :unit:`W`.
    k_fix :
        Fix slope constant.
    alpha_fix :
        Fix exponent constant.

    Returns
    -------
        Fixed power in :unit:`W`.

    """
    return k_fix * p_bad**alpha_fix


def pressure(
    v_acq: NDArray[np.float64], a_calib: float, b_calib: float
) -> NDArray[np.float64]:
    r"""Convert acquisition voltage to pressure.

    This is the transfer function of :class:`.Penning`.

    .. math::
        p = 10^{a_\mathrm{calib} \times V_\mathrm{acq} + b_\mathrm{calib}}

    Parameters
    ----------
    v_acq :
        Acquisition voltage in :math:`[0, 10~\mathrm{V}]`.
    a_calib :
        Calibration slope in :unit:`1/V`.
    b_calib :
        Calibration offset.

    Returns
    -------
        :math:`P` in :unit:`mbar`, which should be the content of the
        ``NI9205_Penningx`` and ``NI9205_bayard-alpert`` columns.

    """
    return np.power(10, a_calib * v_acq + b_calib)
