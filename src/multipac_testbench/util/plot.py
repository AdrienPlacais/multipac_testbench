"""Define helper functions for plots."""

import logging
from abc import ABCMeta
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from multipac_testbench.multipactor_band.test_multipactor_bands import (
    TestMultipactorBands,
)
from multipac_testbench.util.multipactor_detectors import (
    start_and_end_of_contiguous_true_zones,
)
from numpy.typing import NDArray


def create_fig(
    title: str = "",
    instruments_to_plot: Sequence[ABCMeta] = (),
    xlabel: str | None = None,
    subplot_kw: dict | None = None,
    **fig_kw,
) -> tuple[Figure, dict[ABCMeta, Axes]]:
    """Create the figure and axes.

    Parameters
    ----------
    instruments_to_plot :
        Class of the instruments to be plotted.
    fig_kw :
        Keyword arguments passsed to the Figure constructor.

    Returns
    -------
    fig :
        Figure holding the axes.
    instrument_class_axes :
        Dictionary linking the class of the instruments to plot with the
        associated axes.

    """
    fig = plt.figure(**fig_kw)

    if subplot_kw is None:
        subplot_kw = {}
    nrows = len(instruments_to_plot)
    instrument_class_axes = _create_axes(
        instruments_to_plot, fig, nrows, xlabel, **subplot_kw
    )

    if len(title) > 0:
        fig.suptitle(title)
    return fig, instrument_class_axes


def _create_axes(
    instruments_to_plot: Sequence[ABCMeta],
    fig: Figure,
    nrows: int,
    xlabel: str | None = None,
    **subplot_kw,
) -> dict[ABCMeta, Axes]:
    """Create the axes."""
    axes = []
    sharex = None
    for row in range(nrows):
        axe = fig.add_subplot(nrows, 1, row + 1, sharex=sharex, **subplot_kw)
        axes.append(axe)
        sharex = axes[0]

    if xlabel is not None:
        axes[-1].set_xlabel(xlabel)

    instrument_class_axes = dict(zip(instruments_to_plot, axes))

    axe = None
    for instrument_class, axe in instrument_class_axes.items():
        axe.grid(True)
        ylabel = getattr(instrument_class, "ylabel", lambda: "default")()
        axe.set_ylabel(ylabel)
    return instrument_class_axes


def finish_fig(
    fig: Figure,
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


def create_df_to_plot(
    data_to_plot: list[pd.Series],
    tail: int = -1,
    column_names: str | list[str] = "",
    **kwargs,
) -> pd.DataFrame:
    """Merge the series into a single dataframe.

    Parameters
    ----------
    data_to_plot :
        List of the data that will be plotted.
    tail :
        The number of points to plot, starting from the end of the test
        (fully conditioned). The default is ``-1``, in which case the full
        test is plotted.
    column_names :
        To override the default column names. The default is an empty string,
        in which we keep default names. This is used in particular with the
        method :meth:`.TestCampaign.sweet_plot` when ``all_on_same_plot=True``.
    kwargs :
        Other keyword arguments.

    Returns
    -------
    df_to_plot :
        Contains x and y data that will be plotted.

    """
    df_to_plot = pd.concat(data_to_plot, axis=1)
    df_to_plot = df_to_plot.tail(tail)
    # Remove duplicate columns
    df_to_plot = df_to_plot.loc[:, ~df_to_plot.columns.duplicated()].copy()

    if column_names:
        if isinstance(column_names, str):
            column_names = [column_names]
            old_column_names = df_to_plot.columns.values
            assert len(column_names) == len(old_column_names)
            columns_mapper = {
                old: new for old, new in zip(old_column_names, column_names)
            }
            df_to_plot.rename(columns=columns_mapper, inplace=True)

    return df_to_plot


def match_x_and_y_column_names(
    x_columns: list[str] | None,
    y_columns: list[list[str]],
) -> tuple[list[str] | str | None, list[list[str]] | list[str]]:
    """Match name of x columns with y columns, remove duplicate columns.

    Parameters
    ----------
    x_columns :
        Name of the instrument(s) used as x-axis.
    y_columns :
        Name of the instruments for y-axis, sorted by suplot.

    Returns
    -------
    x_columns :
        Name of the instrument(s) used as x-axis.
    y_columns :
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


def actual_plot(
    df_to_plot: pd.DataFrame,
    x_columns: list[str] | str | None,
    y_columns: list[list[str]] | list[str],
    grid: bool = True,
    title: list[str] | str = "",
    sharex: bool | None = True,
    ax: Axes | NDArray[Axes] | None = None,
    color: dict[str, str] | None = None,
    **kwargs,
) -> Axes | NDArray[Axes]:
    """Plot the data, adapting to what is given.

    Parameters
    ----------
    df_to_plot :
        Containts all the data that will be plotted.
    x_columns :
        Name of the column(s) used for x axis.
    y_columns :
        Name of the column(s) for y plot.
    grid :
        If the grid should be plotted.
    title :
        A title for the figure or every subplot if it is a list.
    sharex :
        To let the different subplots share the same x-axis. It is set to None
        when we re-use already existing Axes.
    ax :
        To re-use already existing Axes.
    color :
        Dictionary linking column names in ``df_to_plot`` to HTML colors. Used
        to keep the same color between different instruments at the same
        :class:`.PickUp`.
    kwargs :
        Other keyword arguments passed to the plot function.

    Returns
    -------
    Axes | NDArray[Axes]
        Plotted axes, or an array containing them.

    """
    if ax is not None:
        sharex = None
    if not isinstance(x_columns, list):
        ax = df_to_plot.plot(
            x=x_columns,
            subplots=y_columns,
            sharex=sharex,
            grid=grid,
            title=title,
            ax=ax,
            color=color,
            **kwargs,
        )
        assert ax is not None
        return ax

    zipper = zip(x_columns, y_columns, strict=True)
    for x_col, y_col in zipper:
        ax = df_to_plot.plot(
            x=x_col,
            y=y_col,
            ax=ax,
            grid=grid,
            title=title,
            color=color,
            **kwargs,
        )
    assert ax is not None
    return ax


def set_labels(
    axes: Axes | NDArray[Axes],
    *ydata: ABCMeta,
    xdata: ABCMeta | None = None,
    xlabel: str = "",
    ylabel: str | Iterable = "",
    **kwargs,
) -> None:
    """Set proper ylabel for every subplot.

    Parameters
    ----------
    axes :
        Axes or numpy array containing them.
    *ydata :
        Class of the plotted instruments.
    xdata :
        Class of the x-data instrument if applicable.
    xlabel :
        Label used for x axis. If not given, we take ``ylabel`` attribute
        from ``xdata``.
    ylabel :
        Labels that will be given for every subplot. If not given, we take
        the ``ylabel`` attribute of every plotted class.
    kwargs :
        kwargs

    """
    if not xlabel:
        if xdata is not None:
            xlabel = xdata.ylabel()
        else:
            xlabel = "Sample index"

    if not ylabel:
        ylabel = (obj.ylabel() for obj in ydata)

    if isinstance(ylabel, str):
        ylabel = (ylabel,)
    if isinstance(axes, Axes):
        axes = (axes,)
    for axe, ylab in zip(axes, ylabel):
        axe.set_ylabel(ylab)
        if not xlabel:
            continue
        axe.set_xlabel(xlabel)


def save_figure(
    axes: Axes | NDArray[Axes] | list[Axes],
    png_path: Path,
    verbose: bool = False,
    **png_kwargs,
) -> None:
    """Save the figure.

    Parameters
    ----------
    axes :
        Holds one or several axes.
    png_path :
        Where figure shall be saved.
    verbose :
        To print a message indicating where Figure is saved.
    **png_kwargs :
        Keyword arguments for the ``Figure.savefig`` method.

    """
    if isinstance(axes, (np.ndarray, list)):
        fig = axes[0].get_figure()
    else:
        fig = axes.get_figure()
    assert isinstance(fig, Figure)
    fig.savefig(png_path, **png_kwargs)
    if verbose:
        logging.info(f"Figure saved @ {png_path = }")


def save_dataframe(
    df_to_plot: pd.DataFrame,
    csv_path: Path,
    sep: str = "\t",
    verbose: bool = False,
    **csv_kwargs,
) -> None:
    r"""Save dataframe used to produce the plot.

    Parameters
    ----------
    df_to_plot :
        DataFrame to save.
    csv_path :
        Where to save DataFrame.
    sep :
        Column delimiter.
    verbose :
        To print a message indicating where Figure is saved.
    csv_kwargs :
        Other keyword arguments for the ``DataFrame.to_csv`` method.

    """
    df_to_plot.to_csv(csv_path, sep=sep, **csv_kwargs)
    if verbose:
        logging.info(f"DataFrame saved @ {csv_path = }")


def add_background_color_according_to_power_growth(
    axe: Axes | Sequence[Axes] | NDArray[Axes],
    where_is_growing: list[bool | float],
    grow_kw: dict | None = None,
    decrease_kw: dict | None = None,
    legend: bool = True,
) -> None:
    """Add a background color to indicate where power grows or not.

    Parameters
    ----------
    axe :
        The Axes on which to plot. If several are given, we sequentially call
        this function.
    where_is_growing :
        A list containing True where power grows, False where decreases, np.nan
        when undetermined. Typical return value from
        :meth:`.ForwardPower.where_is_growing`.
    grow_kw :
        How zones where power grows are colored. Default is a semi-transparent
        blue.
    decrease_kw :
        How zones where power decreases are colored. Default is a
        semi-transparent red.
    legend :
        If legend should be added.

    """
    if isinstance(axe, (Sequence, np.ndarray)):
        for ax in axe:
            add_background_color_according_to_power_growth(
                ax, where_is_growing, grow_kw, decrease_kw, legend
            )
        return
    as_array = np.array(where_is_growing)

    if grow_kw is None:
        grow_kw = {"color": "b", "alpha": 0.2}
    _add_single_bg_color(
        as_array, axe, "Power grows", invert_array=False, **grow_kw
    )

    if decrease_kw is None:
        decrease_kw = {"color": "r", "alpha": 0.2}
    _add_single_bg_color(
        as_array, axe, "Power decreases", invert_array=True, **decrease_kw
    )

    if legend:
        axe.legend()


def _add_single_bg_color(
    where_is_growing: NDArray[np.bool],
    axe: Axes,
    label: str | None,
    invert_array: bool,
    **color_kw: dict,
) -> None:
    """Add a single background color to the plot.

    Parameters
    ----------
    where_is_growing :
        Array where 1. means power grows, 0. means it decreases, np.nan is
        undetermined.
    axe :
        Where color should be plotted.
    label :
        The label of the background color.
    invert_array :
        Should be False for grow plot, True for decrease plot. Serve as a
        filling value for nan.
    color_kw :
        Keyword arguments given to axvspan.

    """
    where_is_growing[np.isnan(where_is_growing)] = invert_array
    data = where_is_growing.astype(np.bool_)
    if invert_array:
        data = ~data
    zones = start_and_end_of_contiguous_true_zones(data)
    for zone in zones:
        axe.axvspan(zone[0], zone[1], label=label, **color_kw)
        label = None


def add_instrument_multipactor_bands(
    test_multipactor_bands: TestMultipactorBands,
    axes: NDArray[Axes] | Axes | None = None,
    scale: float = 1.0,
    alpha: float = 0.5,
    legend: bool = True,
    twinx: bool = False,
    **kwargs,
) -> Axes | NDArray[Axes]:
    """Add the multipactor bands to a pre-existing plot."""
    if isinstance(axes, np.ndarray):
        axes_aslist = [
            add_instrument_multipactor_bands(
                test_multipactor_bands,
                axe,
                scale=scale,
                alpha=alpha,
                legend=legend,
                twinx=twinx,
                **kwargs,
            )
            for axe in axes
        ]
        axes = np.array(axes_aslist, dtype=object)
        return axes

    mp_axes = axes
    if twinx:
        assert axes is not None
        mp_axes = axes.twinx()

    mp_axes = test_multipactor_bands.plot_as_bool(
        mp_axes, scale, alpha, legend, **kwargs
    )
    if legend:
        assert axes is not None
        _merge_legends(axes, mp_axes)
    return mp_axes


def _merge_legends(ax1: Axes, ax2: Axes) -> None:
    """Move the legend from ``ax1`` to ``ax2``."""
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend().remove()
    ax2.legend(handles, labels)
