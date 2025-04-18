"""Define various smoothing/smoothing functions for measured data."""

import numpy as np


def running_mean(
    input_data: np.ndarray, n_mean: int, mode: str = "full", **kwargs
) -> np.ndarray:
    """Compute the runnning mean. Taken from `this link`_.

    .. _this link: https://stackoverflow.com/questions/13728392/\
moving-average-or-running-mean

    Parameters
    ----------
    input_data :
        Data to smooth of shape ``N``.
    n_mean :
        Number of points on which running mean is ran.
    mode :
        - By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.
        - 'same': Mode 'same' returns output of length ``max(M, N)``.
          Boundary effects are still visible.
        - 'valid': Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``. The convolution product is only
            given for points where the signals overlap completely. Values
            outside the signal boundary have no effect.
        (taken from numpy documentation)

    Returns
    -------
    data : np.ndarray
        Smoothed data.

    """
    return np.convolve(input_data, np.ones(n_mean) / n_mean, mode=mode)


def v_coax_to_v_acquisition(
    v_coax: np.ndarray,
    g_probe: float,
    a_rack: float,
    b_rack: float,
    z_0: float = 50.0,
) -> np.ndarray:
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
        Line impedance in :math:`\Ohm`.

    Returns
    -------
    v_acq : np.ndarray
        Acquisition voltage in :math:`[0, 10~\mathrm{V}]`.

    """
    p_w = v_coax**2 / (2.0 * z_0)
    p_dbm = 30.0 + 10.0 * np.log10(p_w)
    p_acq = p_dbm - abs(g_probe + 3.0)
    v_acq = (p_acq - b_rack) / a_rack
    return v_acq


def v_acquisition_to_v_coax(
    v_acq: np.ndarray,
    g_probe: float,
    a_rack: float,
    b_rack: float,
    z_0: float = 50.0,
) -> np.ndarray:
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
        Line impedance in :math:`\Ohm`.

    Returns
    -------
    v_coax : np.ndarray
        :math:`V_\mathrm{coax}` in :unit:`V`, which should be the content of
        the ``NI9205_Ex`` columns.

    """
    p_acq = v_acq * a_rack + b_rack
    p_dbm = abs(g_probe + 3.0) + p_acq
    p_w = 10 ** ((p_dbm - 30.0) / 10.0)
    v_coax = np.sqrt(2.0 * z_0 * p_w)
    return v_coax
