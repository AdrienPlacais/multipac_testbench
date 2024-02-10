#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define various smoothing/smoothing functions for measured data."""
import numpy as np


def running_mean(input_data: np.ndarray,
                 n_mean: int,
                 mode: str = 'full',
                 **kwargs) -> np.ndarray:
    """Compute the runnning mean. Taken from `this link`_.

    .. _this link: https://stackoverflow.com/questions/13728392/\
moving-average-or-running-mean

    Parameters
    ----------
    input_data : np.ndarray
        Data to smooth of shape ``N``.
    n_mean : int
        Number of points on which running mean is ran.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.

        'same':
          Mode 'same' returns output of length ``max(M, N)``.  Boundary
          effects are still visible.

        'valid':
          Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.

        (taken from numpy documentation)

    Returns
    -------
    np.ndarray
        Smoothed data.

    """
    return np.convolve(input_data, np.ones(n_mean) / n_mean, mode=mode)


def v_coax_to_v_acquisition(v_coax: np.ndarray,
                            g_probe: float = 1.,
                            a_rack: float = 10.3,
                            b_rack: float = -51.7,
                            z_0: float = 50.,
                            ) -> np.ndarray:
    r"""Convert coaxial voltage to acquisition voltage.

    This is the inverse of the function that is implemented in LabVIEWER.

    Parameters
    ----------
    v_coax : np.ndarray
        :math:`V_\mathrm{coax}` in :math:`\mathrm{V}`, which should be the
        content of the ``NI9205_Ex`` columns.
    g_probe : float, optional
        Total attenuation. Probe specific, also depends on frequency. The
        default is 1., which is the default value when LabVIEWER bugs.
    a_rack : float, optional
        Rack calibration slope in :math:`\mathrm{dBm/V}`. The default
        corresponds to E1 value, which is also the default in LabVIEWER.
    b_rack : float, optional
        Rack calibration constant in :math:`\mathrm{dBm}`. The default
        corresponds to E1 value, which is also the default in LabVIEWER.
    z_0 : float, optional
        Line impedance in :math:`\Ohm`. The default is 50.

    Returns
    -------
    v_acq : np.ndarray
        Acquisition voltage in :math:`[0, 10~\mathrm{V}]`.

    """
    p_w = v_coax**2 / (2. * z_0)
    p_dbm = 30. + 10. * np.log10(p_w)
    p_acq = p_dbm - abs(g_probe + 3.)
    v_acq = (p_acq - b_rack) / a_rack
    return v_acq


def v_acquisition_to_v_coax(v_acq: np.ndarray,
                            g_probe: float = 1.,
                            a_rack: float = 10.3,
                            b_rack: float = -51.7,
                            z_0: float = 50.,
                            ) -> np.ndarray:
    r"""Convert acquisition voltage to coaxial voltage.

    This is the same function that is implemented in LabVIEWER.

    Parameters
    ----------
    v_acq : np.ndarray
        Acquisition voltage in :math:`[0, 10~\mathrm{V}]`.
    g_probe : float, optional
        Total attenuation. Probe specific, also depends on frequency. The
        default is 1., which is the default value when LabVIEWER bugs.
    a_rack : float, optional
        Rack calibration slope in :math:`\mathrm{dBm/V}`. The default
        corresponds to E1 value, which is also the default in LabVIEWER.
    b_rack : float, optional
        Rack calibration constant in :math:`\mathrm{dBm}`. The default
        corresponds to E1 value, which is also the default in LabVIEWER.
    z_0 : float, optional
        Line impedance in :math:`\Ohm`. The default is 50.

    Returns
    -------
    v_coax : np.ndarray
        :math:`V_\mathrm{coax}` in :math:`\mathrm{V}`, which should be the
        content of the ``NI9205_Ex`` columns.

    """
    p_acq = v_acq * a_rack + b_rack
    p_dbm = abs(g_probe + 3.) + p_acq
    p_w = 10**((p_dbm - 30.) / 10.)
    v_coax = np.sqrt(2. * z_0 * p_w)
    return v_coax
