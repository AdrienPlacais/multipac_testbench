#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define helper functions for plots."""
from abc import ABCMeta
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure


def create_fig(title: str = '',
               instruments_to_plot: Sequence[ABCMeta] = (),
               xlabel: str | None = None,
               subplot_kw: dict | None = None,
               **fig_kw,
               ) -> tuple[Figure, dict[ABCMeta, Axes]]:
    """Create the figure and axes.

    Parameters
    ----------
    instruments_to_plot : tuple[ABCMeta, ...]
        Class of the instruments to be plotted.
    fig_kw :
        Keyword arguments passsed to the Figure constructor.

    Returns
    -------
    fig : Figure
        Figure holding the axes.
    instrument_class_axes : dict[ABCMeta, Axes]
        Dictionary linking the class of the instruments to plot with the
        associated axes.

    """
    fig = plt.figure(**fig_kw)

    if subplot_kw is None:
        subplot_kw = {}
    nrows = len(instruments_to_plot)
    instrument_class_axes = _create_axes(instruments_to_plot,
                                         fig,
                                         nrows,
                                         xlabel,
                                         **subplot_kw)

    if len(title) > 0:
        fig.suptitle(title)
    return fig, instrument_class_axes


def _create_axes(instruments_to_plot: Sequence[ABCMeta],
                 fig: Figure,
                 nrows: int,
                 xlabel: str | None = None,
                 **subplot_kw,
                 ) -> dict[ABCMeta, Axes]:
    """Create the axes."""
    axes = []
    sharex = None
    for row in range(nrows):
        axe = fig.add_subplot(nrows, 1, row + 1,
                              sharex=sharex,
                              **subplot_kw)
        axes.append(axe)
        sharex = axes[0]

    if xlabel is not None:
        axes[-1].set_xlabel(xlabel)

    instrument_class_axes = dict(zip(instruments_to_plot, axes))

    axe = None
    for instrument_class, axe in instrument_class_axes.items():
        axe.grid(True)
        ylabel = getattr(instrument_class, 'ylabel', lambda: 'default')()
        axe.set_ylabel(ylabel)
    return instrument_class_axes


def finish_fig(fig: Figure,
               axes: Iterable[Axes],
               png_path: Path | None = None,
               ) -> tuple[Figure, list[Axes]]:
    """Save the figure, create the legend."""
    axes = [axe for axe in axes]
    for axe in axes:
        axe.legend()

    if png_path is not None:
        fig.savefig(png_path)

    return fig, axes
