#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define voltage along line.

.. todo::
    voltage fitting, overload: they work but this not clean, not clean at all

"""
from collections.abc import Sequence
from functools import partial
from typing import overload

import numpy as np
import pandas as pd

from multipac_testbench.src.instruments.electric_field.field_probe import \
    FieldProbe
from multipac_testbench.src.instruments.electric_field.i_electric_field import\
    IElectricField
from multipac_testbench.src.instruments.power import ForwardPower
from multipac_testbench.src.instruments.reflection_coefficient import \
    ReflectionCoefficient
from multipac_testbench.src.util.helper import r_squared
from scipy import optimize
from scipy.constants import c


class Reconstructed(IElectricField):
    """Voltage in the coaxial waveguide, fitted with e field probes."""

    def __init__(self,
                 name: str,
                 raw_data: pd.Series | None,
                 e_field_probes: Sequence[FieldProbe],
                 forward_power: ForwardPower,
                 reflection: ReflectionCoefficient,
                 freq_mhz: float,
                 position: np.ndarray = np.linspace(0., 1.3, 201),
                 z_ohm: float = 50.,
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
        self._forward_power = forward_power
        self._reflection = reflection
        self._sample_indexes = self._e_field_probes[0]._raw_data.index
        self._pos_for_fit = [probe.position for probe in self._e_field_probes]
        self._beta = c / freq_mhz * 1e-6

        self._psi_0: float
        self._data: np.ndarray | None = None
        self._z_ohm = z_ohm
        self._r_squared: float

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Reconstructed voltage [V]"

    @property
    def data(self) -> np.ndarray:
        """Give the calculated voltage at every pos and sample index.

        .. note::
            In contrary to most :class:`Instrument` objects, here ``data`` is
            2D. Axis are the following: ``data[sample_index, position_index]``

        """
        if self._data is not None:
            return self._data

        assert hasattr(self, '_psi_0')

        data = []
        for power, reflection in zip(self._forward_power.data,
                                     self._reflection.data):
            v_f = _power_to_volt(power, z_ohm=self._z_ohm)
            data.append(voltage_vs_position(self.position,
                                            v_f,
                                            reflection,
                                            self._beta,
                                            self._psi_0))
        self._data = np.array(data)
        return self._data

    @property
    def fit_info(self) -> str:
        """Print compact info on fit."""
        out = rf"$\psi_0 = ${self._psi_0:2.3f}"
        if not hasattr(self, '_r_squared'):
            return out

        return '\n'.join([out,
                          rf"$r^2 = ${self._r_squared:2.3f}"])

    @property
    def label(self) -> str:
        """Label used for legends in plots vs position."""
        return self.fit_info

    def fit_voltage(self,
                    full_output: bool = True) -> None:
        r"""Find out the proper voltage parameters.

        Idea is the following: for every sample index we know the forward
        (injected) power :math:`P_f`, :math:`\Gamma`, and
        :math:`V_\mathrm{coax}` at several pick-ups. We try to find
        :math:`\psi_0` to verify:

        .. math::
            |V_\mathrm{coax}(z)| = 2\sqrt{P_f Z} \sqrt{1 + |\Gamma|^2
            + 2|\Gamma| \cos{(2\beta z + \psi_0)}}

        """
        x_0 = np.array([np.pi])
        bounds = ([-2. * np.pi],
                  [2. * np.pi])
        xdata = []
        data = []
        for e_probe in self._e_field_probes:
            for p_f, reflection, e_field in zip(self._forward_power.data,
                                                self._reflection.data,
                                                e_probe.data):
                xdata.append([p_f, reflection, e_probe.position])
                data.append(e_field)

        to_fit = partial(_model, beta=self._beta, z_ohm=self._z_ohm)
        result = optimize.curve_fit(to_fit,
                                    xdata=xdata,  # [power, pos] combinations
                                    ydata=data,  # resulting voltages
                                    p0=x_0,
                                    bounds=bounds,
                                    full_output=full_output,
                                    )
        self._psi_0 = result[0][0]
        if full_output:
            self._r_squared = r_squared(result[2]['fvec'], np.array(data))
            # res_squared = result[2]['fvec']**2
            # expected = np.array(data)

            # ss_err = np.sum(res_squared)
            # ss_tot = np.sum((expected - expected.mean())**2)
            # r_squared = 1. - ss_err / ss_tot
            # self._r_squared = r_squared
            print(self.fit_info)
            print('')

    def _compute_voltages(self,
                          beta: float,
                          psi_0: float,
                          ) -> np.ndarray:
        """Give an the actual voltages we get with given parameters.

        Structure of data is the same as in :attr:`_objective_voltages`.

        """
        actual_voltages = []
        v_f = _power_to_volt(self._forward_power.data, z_ohm=self._z_ohm)
        reflection = self._reflection.data

        for sample_index in self._sample_indexes:
            voltages = voltage_vs_position(self.position,
                                           v_f[sample_index - 1],
                                           reflection[sample_index - 1],
                                           beta,
                                           psi_0,)
            actual_voltages.append(voltages)
        actual_voltages = np.array(actual_voltages)
        return actual_voltages

    def data_at_position(self, pos: float, tol: float = 1e-5) -> np.ndarray:
        """Get reconstructed field at position ``pos``."""
        diff = np.abs(self.position - pos)
        delta_z = np.min(diff)
        if delta_z > tol:
            raise ValueError("You asked for the reconstructed field at "
                             f"position {pos}, but the closest calculated "
                             f"points in {delta_z}m away for it. Check units,"
                             " or increase the number of calculated positions."
                             )
        idx = np.argmin(diff)
        return self.data[:, idx]


def _model(var: np.ndarray,
           psi_0: float,
           beta: float,
           z_ohm: float = 50.,
           ) -> float:
    r"""Give voltage for given set of parameters, at proper power and position.

    Parameters
    ----------
    var : np.ndarray
        Variables, namely :math:`[P_f, R, z]`.

    Returns
    -------
    v : float | np.ndarray
        Voltage at position :math:`z` for forward power :math:`P_f`.

    """
    power, reflection, pos = var[:, 0], var[:, 1], var[:, 2]
    v_f = _power_to_volt(power, z_ohm=z_ohm)
    return voltage_vs_position(pos, v_f, reflection, beta, psi_0)


def _power_to_volt(power: np.ndarray,
                   z_ohm: float = 50.) -> np.ndarray:
    return 2. * np.sqrt(power * z_ohm)


@overload
def voltage_vs_position(pos: float,
                        v_f: float,
                        reflection: float,
                        beta: float,
                        psi_0: float,
                        ) -> float: ...


@overload
def voltage_vs_position(pos: np.ndarray,
                        v_f: float,
                        reflection: float,
                        beta: float,
                        psi_0: float,
                        ) -> np.ndarray: ...


def voltage_vs_position(pos: float | np.ndarray,
                        v_f: float,
                        reflection: float,
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
    assert not isinstance(reflection, complex), "not implemented"

    voltage = v_f * np.sqrt(
        1. + reflection**2 + 2. * reflection * np.cos(2. * beta * pos + psi_0)
    )
    return voltage
