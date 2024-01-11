#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define voltage along line.

.. todo::
    Implement a way to get reconstructed voltage at given pick-up and sample
    index.

.. todo::
    voltage fitting, overload: they work but this not clean, not clean at all

"""
import math
from functools import partial
from typing import overload

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.constants import c

from multipac_testbench.src.instruments.electric_field.field_probe import \
    FieldProbe
from multipac_testbench.src.instruments.electric_field.i_electric_field import \
    IElectricField
from multipac_testbench.src.instruments.powers import Powers


class Reconstructed(IElectricField):
    """Voltage in the coaxial waveguide, fitted with e field probes."""

    def __init__(self,
                 name: str,
                 raw_data: pd.Series | None,
                 e_field_probes: list[FieldProbe],
                 powers: Powers,
                 freq_mhz: float,
                 position: np.ndarray = np.linspace(0., 1.3, 201),
                 **kwargs,
                 ) -> None:
        """Just instantiate."""
        # from_array maybe
        super().__init__(name,
                         raw_data,
                         position=position,
                         is_2d=True,
                         **kwargs)
        self._e_field_probes = e_field_probes
        self._powers = powers
        self._sample_indexes = self._e_field_probes[0].raw_data.index
        self._pos_for_fit = [probe._position for probe in self._e_field_probes]
        self._beta = c / freq_mhz * 1e-6

        self._sqrt_power_scaler: float
        self._psi_0: float
        self._ydata: np.ndarray | None = None

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Reconstructed voltage [V]"

    @property
    def ydata(self) -> np.ndarray:
        """Give the calculated voltage at every pos and sample index.

        .. note::
            In contrary to most :class:`Instrument` objects, here ``ydata`` is
            2D. Axis are the following: ``ydata[sample_index, position_index]``
            .

        """
        if self._ydata is not None:
            return self._ydata

        mandatory_args = ('_sqrt_power_scaler', '_psi_0')
        for arg in mandatory_args:
            assert hasattr(self, arg)

        ydata = []
        for power, gamma in zip(self._powers.ydata[:, 0], self._powers.gamma):
            v_f = math.sqrt(power) * self._sqrt_power_scaler
            ydata.append(voltage_vs_position(self._position,
                                             v_f,
                                             gamma,
                                             self._beta,
                                             self._psi_0))
        self._ydata = np.array(ydata)
        return self._ydata

    @property
    def fit_info(self) -> str:
        """Print compact info on fit."""
        out = [f"$k = ${self._sqrt_power_scaler:2.3f}",
               rf"$\psi_0 = ${self._psi_0:2.3f}",
               ]
        return '\n'.join(out)

    @property
    def label(self) -> str:
        """Label used for legends in plots vs position."""
        return self.fit_info

    def fit_voltage(self) -> None:
        r"""Find out the proper voltage parameters.

        Idea is the following: for every sample index we know the forward
        (injected) power :math:`P_f`, :math:`\Gamma`, and
        :math:`V_\mathrm{coax}` at several pick-ups. We try to find :math:`k`
        and :math:`\psi_0` to verify:

        .. math::
            |V_\mathrm{coax}(z)| = k\sqrt{P_f} \sqrt{1 + |\Gamma|^2 + 2|\Gamma|
            \cos{(2\beta z + \psi_0)}}

        """
        x_0 = np.array([0.003, np.pi])
        bounds = ([0., -2. * np.pi],
                  [np.inf, 2. * np.pi])
        xdata = []
        ydata = []
        for e_probe in self._e_field_probes:
            for p_f, gamma, e_field in zip(self._powers.ydata[:, 0],
                                           self._powers.gamma,
                                           e_probe.ydata):
                xdata.append([p_f, gamma, e_probe._position])
                ydata.append(e_field)

        to_fit = partial(_model, beta=self._beta)
        result = optimize.curve_fit(to_fit,
                                    xdata=xdata,  # [power, pos] combinations
                                    ydata=ydata,  # resulting voltages
                                    p0=x_0,
                                    bounds=bounds,
                                    )
        sqrt_power_scaler, psi_0 = result[0]
        self._sqrt_power_scaler = sqrt_power_scaler
        self._psi_0 = psi_0

    def _compute_voltages(self,
                          sqrt_power_scaler: float,
                          beta: float,
                          psi_0: float
                          ) -> np.ndarray:
        """Give an the actual voltages we get with given parameters.

        Structure of data is the same as in :attr:`_objective_voltages`.

        """
        actual_voltages = []
        v_f = sqrt_power_scaler * np.sqrt(self._powers.ydata[:, 0])
        gamma = self._powers.gamma

        for sample_index in self._sample_indexes:
            voltages = voltage_vs_position(self._position,
                                           v_f[sample_index - 1],
                                           gamma[sample_index - 1],
                                           beta,
                                           psi_0)
            actual_voltages.append(voltages)
        actual_voltages = np.array(actual_voltages)
        return actual_voltages


def _model(var: np.ndarray,
           sqrt_power_scaler: float,
           psi_0: float,
           beta: float
           ) -> float:
    r"""Give voltage for given set of parameters, at proper power and position.

    Parameters
    ----------
    var : np.ndarray
        Variables, namely :math:`[P_f, \Gamma, z]`.

    Returns
    -------
    v : float | np.ndarray
        Voltage at position :math:`z` for forward power :math:`P_f`.

    """
    power, gamma, pos = var[:, 0], var[:, 1], var[:, 2]
    v_f = sqrt_power_scaler * np.sqrt(power)
    return voltage_vs_position(pos, v_f, gamma, beta, psi_0)


@overload
def voltage_vs_position(pos: float,
                        v_f: float,
                        gamma: float,
                        beta: float,
                        psi_0: float,
                        ) -> float: ...


@overload
def voltage_vs_position(pos: np.ndarray,
                        v_f: float,
                        gamma: float,
                        beta: float,
                        psi_0: float,
                        ) -> np.ndarray: ...


def voltage_vs_position(pos: float | np.ndarray,
                        v_f: float,
                        gamma: float,
                        beta: float,
                        psi_0: float,
                        ) -> float | np.ndarray:
    r"""Compute voltage in coaxial line at given position.

    The equation is:

    .. math::
        |V(z)| = |V_f| \sqrt{1 + |\Gamma|^2 + 2|\Gamma|\cos{(2\beta z +
        \psi_0)}}

    which comes from:

    .. math::
        V(z) = V_f \mathrm{e}^{-j\beta z} + \Gamma V_f \mathrm{e}^{j\beta z}

    Parameters
    ----------
    pos : float | np.ndarray
        :math:`z` position in :math:`m`.
    v_f : float
        Forward voltage :math:`V_f` in :math:`V`.
    gamma : float
        Voltage reflexion coefficient :math:`\Gamma`.
    beta : float
        Propagation constant :math:`\beta` in :math:`m^{-1}`.
    psi_0 : float
        Dephasing constant :math:`\psi_0`.

    Returns
    -------
    float | np.ndarray
        :math:`V(z)` at proper position in :math:`V`.

    """
    assert not isinstance(v_f, complex), "not implemented"
    assert not isinstance(gamma, complex), "not implemented"

    voltage = v_f * np.sqrt(
        1. + gamma**2 + 2. * gamma * np.cos(2. * beta * pos + psi_0)
    )
    return voltage
