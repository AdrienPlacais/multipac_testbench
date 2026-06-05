"""
Interactive fig coupling :class:`.MultipactorTest` with :class:`.PowerStep`.
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import matplotlib
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, KeyEvent, MouseEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from multipac_testbench.instruments.instrument import Instrument

if TYPE_CHECKING:
    from multipac_testbench.multipactor_test.multipactor_test import (
        MultipactorTest,
    )
    from multipac_testbench.multipactor_test.power_step import PowerStep


class InteractivePlot:
    """Couple :class:`.MultipactorTest` plot with :class:`.PowerStep` plot.

    Click on the overview to show the corresponding :class:`.PowerStep`.
    Navigate between steps with the left/right arrow keys.
    Toggle raw/physical data with the button.

    """

    def __init__(
        self,
        test: MultipactorTest,
        ydata: tuple[ABCMeta, ...],
        xdata: ABCMeta | None,
        kwargs: dict[str, Any],
    ) -> None:
        """Instantiate object."""
        self._test = test
        _power_step_set = test.power_step_set
        assert _power_step_set is not None
        self._power_step_set = _power_step_set
        self._ydata: list[ABCMeta] = list(ydata)
        self._xdata = xdata
        self._kwargs = kwargs

        self._raw: bool = False
        self._sample_index: int | None = None

        self._axes: list[Axes] = []
        self._vlines: list[Line2D] = []
        self._ps_axes: list[Axes] | None = None
        self._ps_fig: Figure | None = None
        self._toggle_btn: Button | None = None

    def show(self) -> tuple[list[Axes], pd.DataFrame]:
        """Create the overview figure and wire all event handlers.

        Returns
        -------
            Same as :meth:`.MultipactorTest.sweet_plot`.

        """
        axes, df = self._test.sweet_plot(
            *self._ydata, xdata=self._xdata, raw=False, **self._kwargs
        )
        self._axes = axes

        self._fig.subplots_adjust(bottom=0.12)
        btn_ax = self._fig.add_axes((0.45, 0.02, 0.12, 0.04))
        self._toggle_btn = Button(btn_ax, "Show raw")
        self._toggle_btn.on_clicked(self._on_toggle)

        self._vlines = [
            ax.axvline(x=0, color="gray", lw=0.8, ls="--", visible=False)
            for ax in axes
        ]

        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect(
            "button_press_event", self._on_mouse_press
        )
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        return axes, df

    @property
    def _fig(self) -> Figure:
        """Get main figure."""
        fig = self._axes[0].get_figure()
        assert isinstance(fig, Figure)
        return fig

    @cached_property
    def _plottable(self) -> list[type[Instrument]]:
        """Unique instrument classes present in the test, preserving order."""
        seen: set[type[Instrument]] = set()
        result: list[type[Instrument]] = []
        for instrument in self._test.get_instruments(Instrument):
            cls = type(instrument)
            if cls not in seen:
                seen.add(cls)
                result.append(cls)
        return result

    def _on_mouse_press(self, event: Event) -> None:
        """Left-click draws a PowerStep; right-click opens instrument menu."""
        assert isinstance(event, MouseEvent)
        if event.inaxes not in self._axes:
            return
        if event.button == 1:
            self._draw_power_step(round(event.xdata))  # type: ignore[arg-type]
        elif event.button == 3:
            self._show_instrument_menu(event, self._axes.index(event.inaxes))

    def _on_key(self, event: Event) -> None:
        """Update :class:`.PowerStep` plot when hitting left/right."""
        assert isinstance(event, KeyEvent)
        if event.key not in ("left", "right") or self._sample_index is None:
            return
        delta = -1 if event.key == "left" else 1
        next_index = self._sample_index + delta
        try:
            self._power_step_set.get_power_step(next_index)
        except KeyError:
            return
        self._draw_power_step(next_index)

    def _on_motion(self, event: Event) -> None:
        """Draw vertical line on main plot below mouse."""
        assert isinstance(event, MouseEvent)
        if event.inaxes not in self._axes:
            for vl in self._vlines:
                vl.set_visible(False)
        else:
            for vl in self._vlines:
                vl.set_visible(True)
                vl.set_xdata([round(event.xdata)])  # type: ignore[arg-type]
        self._fig.canvas.draw_idle()

    def _on_toggle(self, _: Event) -> None:
        """Toggle between raw/physical when button is pressed."""
        self._raw = not self._raw
        assert self._toggle_btn is not None
        self._toggle_btn.label.set_text(
            "Show physical" if self._raw else "Show raw"
        )
        self._redraw()

    def _redraw(self) -> None:
        """Redraw the overview after a state change (e.g. raw toggle)."""
        for ax in self._axes:
            ax.cla()
        self._test.sweet_plot(
            *self._ydata,
            xdata=self._xdata,
            axes=self._axes,
            raw=self._raw,
            **self._kwargs,
        )
        self._vlines.clear()
        self._vlines.extend(
            ax.axvline(x=0, color="gray", lw=0.8, ls="--", visible=False)
            for ax in self._axes
        )
        if self._sample_index is not None:
            for vl in self._vlines:
                vl.set_visible(True)
                vl.set_xdata([self._sample_index])
            self._draw_power_step(self._sample_index)
        self._fig.canvas.draw_idle()

    def _draw_power_step(self, sample_index: int) -> None:
        """Show or update the PowerStep figure for ``sample_index``."""
        try:
            power_step = self._power_step_set.get_power_step(sample_index)
        except KeyError:
            logging.warning(f"No PowerStep found for {sample_index = }.")
            return
        self._sample_index = sample_index

        pre_trig = power_step.test_conditions.pre_trigger
        trig = power_step.test_conditions.trigger

        if self._ps_fig is None:
            ps_axes, _ = power_step.sweet_plot(
                *self._ydata, raw=self._raw, pre_trig=pre_trig, trig=trig
            )
            self._ps_axes = ps_axes
            ps_fig = ps_axes[0].get_figure()
            assert isinstance(ps_fig, Figure)
            self._ps_fig = ps_fig
            self._ps_fig.canvas.mpl_connect("key_press_event", self._on_key)
            self._ps_fig.show()
        else:
            assert self._ps_axes is not None
            for ax in self._ps_axes:
                ax.cla()
            power_step.sweet_plot(
                *self._ydata,
                axes=self._ps_axes,
                raw=self._raw,
                pre_trig=pre_trig,
                trig=trig,
            )
            self._ps_fig.canvas.draw_idle()

        self._annotate_reduction_info(self._ps_axes, power_step)

        for vl in self._vlines:
            vl.set_visible(True)
            vl.set_xdata([sample_index])
        self._fig.canvas.draw_idle()

    def _annotate_reduction_info(
        self, ps_axes: list[Axes] | None, power_step: PowerStep
    ) -> None:
        """Append :attr:`.Instrument.reduction_info` to each legend label."""
        if ps_axes is None:
            logging.error("Cannot annotate non-existing plot.")
            return
        info_by_name = {
            instrument.name: instrument.reduction_info
            for instrument in power_step.get_instruments(Instrument)
            if instrument.reduction_info is not None
        }
        for ax in ps_axes:
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                continue
            new_labels = [
                (
                    f"{label}\n{info_by_name[label]}"
                    if label in info_by_name
                    else label
                )
                for label in labels
            ]
            ax.legend(handles, new_labels)

    def _show_instrument_menu(self, event: MouseEvent, ax_index: int) -> None:
        """Show a Qt context menu to pick the instrument type for ``ax_index``."""
        if not self._plottable:
            return

        if "qt" not in matplotlib.get_backend().lower():
            logging.warning(
                "Right-click instrument selection requires a Qt backend. "
                f"Current backend: {matplotlib.get_backend()}. Add "
                "`matplotlib.use('QtAgg')` before importing matplotlib to "
                "enable this feature."
            )
            return
        from matplotlib.backends.qt_compat import QtCore  # type: ignore[attr-defined]
        from matplotlib.backends.qt_compat import (
            QtWidgets,  # type: ignore[attr-defined]
        )

        menu = QtWidgets.QMenu()
        current_cls = self._ydata[ax_index]
        for cls in self._plottable:
            action = menu.addAction(cls.__name__)
            action.setCheckable(True)
            action.setChecked(cls is current_cls)
            action.triggered.connect(
                lambda _, c=cls, i=ax_index: self._set_ydata(i, c)
            )

        canvas = self._fig.canvas
        qt_canvas = cast(Any, canvas)
        qt_pos = qt_canvas.mapToGlobal(
            QtCore.QPoint(int(event.x), qt_canvas.height() - int(event.y))
        )
        menu.exec(qt_pos)

    def _set_ydata(self, ax_index: int, cls: type[Instrument]) -> None:
        """Update the instrument type for subplot ``ax_index`` and redraw."""
        if cls is self._ydata[ax_index]:
            return
        self._ydata[ax_index] = cls
        self._redraw()
