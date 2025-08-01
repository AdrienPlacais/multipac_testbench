"""Define theoretical voltage along line.

Also define a virtual instrument to quantify error between theoretical voltage
and voltage measured by probes.

.. todo::
    voltage fitting, overload: they work but this not clean, not clean at all

"""

import logging
from collections.abc import Sequence
from functools import partial
from typing import Self, overload

import numpy as np
import pandas as pd
from multipac_testbench.instruments import FieldProbe
from multipac_testbench.instruments.electric_field.field_probe import (
    FieldProbe,
)
from multipac_testbench.instruments.electric_field.i_electric_field import (
    IElectricField,
)
from multipac_testbench.instruments.power import ForwardPower
from multipac_testbench.instruments.reflection_coefficient import (
    ReflectionCoefficient,
)
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from multipac_testbench.util.helper import r_squared
from numpy.typing import NDArray
from scipy import optimize
from scipy.constants import c


class Reconstructed(IElectricField):
    """Voltage in the coaxial waveguide, fitted with e field probes."""

    def __init__(
        self,
        name: str,
        raw_data: pd.Series | None,
        e_field_probes: Sequence[FieldProbe],
        forward_power: ForwardPower,
        reflection: ReflectionCoefficient,
        freq_mhz: float,
        position: NDArray[np.float64] | None = None,
        z_ohm: float = 50.0,
        **kwargs,
    ) -> None:
        """Just instantiate.

        Created by :meth:`.MultipactorTest.reconstruct_voltage_along_line`.

        """
        if position is None:
            position = np.linspace(0.0, 1.3, 201, dtype=np.float64)
        # from_array maybe
        super().__init__(
            name, raw_data, position=position, is_2d=True, **kwargs
        )
        self._e_field_probes = e_field_probes
        self._forward_power = forward_power
        self._reflection = reflection
        self._sample_indexes = self._e_field_probes[0]._raw_data.index
        self._beta = c / freq_mhz * 1e-6

        self._psi_0: float
        self._data: NDArray[np.float64] | None = None
        self._z_ohm = z_ohm
        self._r_squared: float

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Reconstructed voltage [V]"

    @property
    def data(self) -> NDArray[np.float64]:
        """Give the calculated voltage at every pos and sample index.

        .. note::
            In contrary to most :class:`.Instrument` objects, here ``data`` is
            2D. Axis are the following: ``data[sample_index, position_index]``

        """
        if self._data is not None:
            return self._data

        assert hasattr(self, "_psi_0")

        data = []
        for power, reflection in zip(
            self._forward_power.data, self._reflection.data
        ):
            v_f = _power_to_volt(power, z_ohm=self._z_ohm)
            data.append(
                voltage_vs_position(
                    self.position, v_f, reflection, self._beta, self._psi_0
                )
            )
        self._data = np.array(data)
        return self._data

    @property
    def fit_info(self) -> str:
        """Print compact info on fit."""
        out = rf"$\psi_0 = ${self._psi_0:2.3f}"
        if not hasattr(self, "_r_squared"):
            return out

        return "\n".join([out, rf"$r^2 = ${self._r_squared:2.3f}"])

    @property
    def label(self) -> str:
        """Label used for legends in plots vs position."""
        return self.fit_info

    def fit_voltage(self, full_output: bool = True) -> None:
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
        bounds = ([-2.0 * np.pi], [2.0 * np.pi])
        xdata = []
        data = []
        _prev = 0.0
        for e_probe in self._e_field_probes:
            for p_f, reflection, e_field in zip(
                self._forward_power.data, self._reflection.data, e_probe.data
            ):
                if np.isnan(reflection):
                    logging.warning(
                        "NaN value found in Reflection. Dirty patch: setting "
                        "Reflection to it's value in previous time step."
                    )
                    reflection = _prev
                xdata.append([p_f, reflection, e_probe.position])
                data.append(e_field)
                _prev = reflection

        to_fit = partial(_model, beta=self._beta, z_ohm=self._z_ohm)
        result = optimize.curve_fit(
            to_fit,
            xdata=xdata,  # [power, pos] combinations
            ydata=data,  # resulting voltages
            p0=x_0,
            bounds=bounds,
            full_output=full_output,
        )
        self._psi_0 = result[0][0]
        if full_output:
            self._r_squared = r_squared(result[2]["fvec"], np.array(data))
            # res_squared = result[2]['fvec']**2
            # expected = np.array(data)

            # ss_err = np.sum(res_squared)
            # ss_tot = np.sum((expected - expected.mean())**2)
            # r_squared = 1. - ss_err / ss_tot
            # self._r_squared = r_squared
            logging.debug(self.fit_info)

    def data_at(self, position: float) -> NDArray[np.float64]:
        """Evaluate evolution of voltage during test at a specific position."""
        var = np.column_stack(
            [
                self._forward_power.data,
                self._reflection.data,
                np.full_like(self._reflection.data, position),
            ]
        )
        data = _model(
            var,
            self._psi_0,
            self._beta,
            self._z_ohm,
        )

        return data


def _model(
    var: NDArray[np.float64],
    psi_0: float,
    beta: float,
    z_ohm: float = 50.0,
) -> NDArray[np.float64]:
    r"""Give voltage for given set of parameters, at proper power and position.

    Parameters
    ----------
    var :
        2D array holding variables in it's three columns: :math:`[P_f, R, z]`.

    Returns
    -------
        Voltage at position :math:`z` for forward power :math:`P_f`.

    """
    power, reflection, pos = var[:, 0], var[:, 1], var[:, 2]
    v_f = _power_to_volt(power, z_ohm=z_ohm)
    return voltage_vs_position(pos, v_f, reflection, beta, psi_0)


def _power_to_volt(
    power: NDArray[np.float64], z_ohm: float = 50.0
) -> NDArray[np.float64]:
    r"""Convert power in :unit:`W` to voltage in :unit:`V`.

    Parameters
    ----------
    power :
        Power in :unit:`W`.
    z_ohm :
        Line impedance in :unit:`\\Omega`.

    Returns
    -------
        Voltage in :unit:`V`.

    """
    return 2.0 * np.sqrt(power * z_ohm)


@overload
def voltage_vs_position(
    pos: float,
    v_f: float,
    reflection: float,
    beta: float,
    psi_0: float,
) -> float: ...


@overload
def voltage_vs_position(
    pos: NDArray[np.float64],
    v_f: float,
    reflection: float,
    beta: float,
    psi_0: float,
) -> NDArray[np.float64]: ...


def voltage_vs_position(
    pos: float | NDArray[np.float64],
    v_f: float,
    reflection: float,
    beta: float,
    psi_0: float,
) -> float | NDArray[np.float64]:
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
    pos :
        :math:`z` position in :unit:`m`.
    v_f :
        Forward voltage :math:`V_f` in :unit:`V`.
    gamma :
        Voltage reflexion coefficient :math:`\Gamma`.
    beta :
        Propagation constant :math:`\beta` in :unit:`m^{-1}`.
    psi_0 :
        Dephasing constant :math:`\psi_0`.

    Returns
    -------
        :math:`V(z)` at proper position in :unit:`V`.

    """
    assert not isinstance(v_f, complex), "not implemented"
    assert not isinstance(reflection, complex), "not implemented"

    voltage = v_f * np.sqrt(
        1.0
        + reflection**2
        + 2.0 * reflection * np.cos(2.0 * beta * pos + psi_0)
    )
    return voltage


class FieldPowerError(VirtualInstrument):
    r"""Store relative error between a field probe and theoretical voltage.

    It is defined for every field probe, at a specific position.

    .. math::

        \epsilon_V = \frac{V_{theoretical} - V_{probe}}{V_{theoretical}}

    """

    def __init__(
        self,
        name: str,
        raw_data: pd.Series,
        reconstructed: Reconstructed,
        field_probe: FieldProbe,
        rmse: float,
        **kwargs,
    ) -> None:
        """Compute error.

        Instantiated when a :class:`.Reconstructed` is created (ie, when
        :meth:`.MultipactorTest.reconstruct_voltage_along_line` is called).

        """
        super().__init__(name, raw_data, **kwargs)
        self._reconstructed = reconstructed
        self._field_probe = field_probe
        self.rmse = rmse
        self.name += f" RMSE: {self.rmse:.2f}V"

    @property
    def data(self) -> NDArray[np.float64]:
        """Get error data."""
        if (data := getattr(self, "_data", None)) is not None:
            return data
        error, self.rmse = compute_field_errors(
            self._reconstructed, self._field_probe
        )
        self._data = np.array(error)
        return self._data

    @data.setter
    def data(self, new_data: NDArray[np.float64]) -> None:
        """Set ``data``, clean previous ``_data_as_pd``."""
        self._data = new_data
        if hasattr(self, "_data_as_pd"):
            delattr(self, "_data_as_pd")

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Field relative error [%]"

    @classmethod
    def from_instruments(
        cls, reconstructed: Reconstructed, field_probe: FieldProbe, **kwargs
    ) -> Self:
        """Create object from other instruments."""
        error, rmse = compute_field_errors(reconstructed, field_probe)
        return cls(
            name=f"Error at {field_probe.name}",
            raw_data=error,
            rmse=rmse,
            position=field_probe.position,
            is_2d=False,
            color=field_probe.color,
            reconstructed=reconstructed,
            field_probe=field_probe,
            **kwargs,
        )


def compute_field_errors(
    reconstructed: Reconstructed, field_probe: FieldProbe
) -> tuple[pd.Series, float]:
    """Compute relative error between the two instruments.

    The theoretical electric field (calculated from measured forward and
    reflected powers) is taken as reference.

    Also compute Root Mean Square Error.

    """
    position = field_probe.position
    assert isinstance(position, float)
    theoretical = reconstructed.data_at(position)
    error = 100.0 * np.abs((theoretical - field_probe.data) / theoretical)
    rmse = np.sqrt(np.mean((theoretical - field_probe.data) ** 2))
    return pd.Series(error), float(rmse)
