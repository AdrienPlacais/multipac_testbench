#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define helper functions for plots."""
from abc import ABCMeta
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
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


def create_df_to_plot(data_to_plot: list[pd.Series],
                      tail: int = -1,
                      **kwargs,
                      ) -> pd.DataFrame:
    """Merge the series into a single dataframe.

    Parameters
    ----------
    data_to_plot : list[pd.Series]
        List of the data that will be plotted.
    tail : int, optional
        The number of points to plot, starting from the end of the test
        (fully conditioned). The default is ``-1``, in which case the full
        test is plotted.
    kwargs :
        Other keyword arguments.

    Returns
    -------
    df_to_plot : pd.DataFrame
        Contains x and y data that will be plotted.

    """
    df_to_plot = pd.concat(data_to_plot, axis=1)
    df_to_plot = df_to_plot.tail(tail)
    # Remove duplicate columns
    df_to_plot = df_to_plot.loc[:, ~df_to_plot.columns.duplicated()].copy()
    return df_to_plot


def match_x_and_y_column_names(
        x_columns: list[str] | None,
        y_columns: list[list[str]],
        ) -> tuple[list[str] | str | None, list[list[str]] | list[str]]:
    """Match name of x columns with y columns, remove duplicate columns.

    Parameters
    ----------
    x_columns : list[str] | None
        Name of the instrument(s) used as x-axis.
    y_columns : list[list[str]]
        Name of the instruments for y-axis, sorted by suplot.

    Returns
    -------
    x_columns : list[str] | str | None
        Name of the instrument(s) used as x-axis.
    y_columns : list[list[str]] | list[str]
        Name of the instruments for y-axis.

    """
    # One or several instrument types plotted vs Sample index
    if x_columns is None:
        return x_columns, y_columns

    # One or several instruments types plotted vs another single instrument
    if len(x_columns) == 1:
        x_column = x_columns[0]

        for y_column in y_columns:
            if x_column in y_column:
                y_column.remove(x_column)

        return x_column, y_columns

    # One instrument type plotted vs another instrument type
    # number of instruments should match
    x_column = x_columns
    y_column = y_columns[0]
    return x_column, y_column


def actual_plot(df_to_plot: pd.DataFrame,
                x_columns: list[str] | str | None,
                y_columns: list[list[str]] | list[str],
                grid: bool = True,
                title: list[str] | str = '',
                **kwargs) -> Axes | np.ndarray[Axes]:
    """Plot the data, adapting to what is given.

    Parameters
    ----------
    df_to_plot : pd.DataFrame
        Containts all the data that will be plotted.
    x_columns : list[str] | None
        Name of the column(s) used for x axis.
    y_columns : list[list[str]]
        Name of the column(s) for y plot.
    grid : bool, optional
        If the grid should be plotted. The default is True.
    title : list[str] | str, optional
        A title for the figure or every subplot if it is a list. The
        default is an empty string.
    kwargs :
        Other keyword arguments passed to the plot function.

    Returns
    -------
    Axes | np.ndarray[Axes]
        Plotted axes, or an array containing them.

    """
    if not isinstance(x_columns, list):
        axes = df_to_plot.plot(x=x_columns,
                               subplots=y_columns,
                               sharex=True,
                               grid=grid,
                               title=title,
                               **kwargs)
        return axes

    axes = None
    zipper = zip(x_columns, y_columns, strict=True)
    for x_col, y_col in zipper:
        axes = df_to_plot.plot(x=x_col,
                               y=y_col,
                               ax=axes,
                               grid=grid,
                               title=title,
                               **kwargs)
    assert axes is not None
    return axes


def set_labels(axes: Axes | np.ndarray[Axes],
               *ydata: ABCMeta,
               xdata: ABCMeta | None = None,
               xlabel: str = '',
               ylabel: str | Iterable = '',
               **kwargs
               ) -> None:
    """Set proper ylabel for every subplot.

    Parameters
    ----------
    axes : Axes | np.ndarray[Axes]
        Axes or numpy array containing them.
    *ydata : ABCMeta
        Class of the plotted instruments.
    xdata : ABCMeta | None, optional
        Class of the x-data instrument if applicable. The default is None.
    xlabel : str, optional
        Label used for x axis. If not given, we take ``ylabel`` attribute
        from ``xdata``.
    ylabel : str | Iterable, optional
        Labels that will be given for every subplot. If not given, we take
        the ``ylabel`` attribute of every plotted class.
    kwargs :
        kwargs

    """
    if not xlabel and xdata is not None:
        xlabel = xdata.ylabel()

    if not ylabel:
        ylabel = (obj.ylabel() for obj in ydata)

    if isinstance(ylabel, str):
        ylabel = ylabel,
    if isinstance(axes, Axes):
        axes = axes,
    for axe, ylab in zip(axes, ylabel):
        axe.set_ylabel(ylab)
        if not xlabel:
            continue
        axe.set_xlabel(xlabel)


def save_figure(axes: Axes | np.ndarray[Axes] | list[Axes],
                png_path: Path,
                **png_kwargs) -> None:
    """Save the figure.

    Parameters
    ----------
    axes : Axes | np.ndarray[Axes]
        Holds one or several axes.
    png_path : Path
        Where figure shall be saved.
    **png_kwargs :
        Keyword arguments for the ``savefig`` method.

    """
    if isinstance(axes, (np.ndarray, list)):
        fig = axes[0].get_figure()
    else:
        fig = axes.get_figure()
    fig.savefig(png_path, **png_kwargs)


def save_dataframe(df_to_plot: pd.DataFrame,
                   csv_path: Path,
                   sep: str = '\t',
                   **csv_kwargs) -> None:
    """Save dataframe used to produce the plot.

    Parameters
    ----------
    df_to_plot : pd.DataFrame
        DataFrame to save.
    csv_path : Path
        Where to save DataFrame.
    sep : str, optional
        Column delimiter. The default is ``'\t'``.
    csv_kwargs :
        Othe keyword arguments.

    """
    df_to_plot.to_csv(csv_path, sep=sep, **csv_kwargs)
