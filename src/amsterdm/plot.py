"""Plotting functions for quick analysis plots

These functions provide basic plots for analysis of FRBs, such as a
waterfall plot (dynamical spectrum), a light curve, a "bowtie" plot, a
(peak) signal to noise graph, and an all-in-one plot

While some options can be set through keyword arguments, the functions
aim to provide only basic functionality; for publication-level
figures, one will likely want to create their own figures manually.

"""

# To do: change `burst` argument to data, freqs, tsamp. With the Burst
# class now have plotting methods, this can be replaced by more
# straightforward arguments in the plotting functions, which then become
# more "fundamental". This will also remove the local "burst" imports,
# which are currently necessary to avoid circular imports, as well as
# the use of TYPE_CECKING for importing Burst for type-checking.
#
# Note that the use of a secondary axis (requiring burst.freq2channel)
# may make it more difficult to get rid of Burst instances.

from __future__ import annotations  # for Burst type

import logging
from types import EllipsisType

from astropy.time import Time
from astropy.units import Quantity
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .constants import DEFAULT_BACKGROUND_RANGE, DMCONST, DMUNIT, Array
from . import core
from .utils import FInterval, symlog


logger = logging.getLogger(__package__)


def ensure_figure(
    ax: Axes | None, figsize: tuple[float, float] = (12, 8)
) -> tuple[Figure, Axes]:
    if not ax:
        # Create a new figure
        # Use the pyplot interface for automatic
        # visualization in notebooks
        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot()
    else:
        figure = ax.figure
    return figure, ax


def set_title_labels(kwargs, defaults, ax):

    title = kwargs.get("title", defaults.get("title", ""))
    if title:
        ax.set_title(title)
    xlabel = kwargs.get("xlabel", defaults.get("xlabel", ""))
    if xlabel:
        ax.set_xlabel(xlabel)
    ylabel = kwargs.get("ylabel", defaults.get("ylabel", ""))
    if ylabel:
        ax.set_ylabel(ylabel)


def waterfall(
    data: Array,
    freqs: np.ndarray,
    tsamp: float,
    dm: float | Quantity = 0,
    reffreq: float | None = None,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
    background: tuple[float, float] | None = None,
    return_image: bool = False,
    ax: Axes | None = None,
    **options,
) -> tuple[Figure, Axes] | tuple[tuple[Figure, Axes], None]:
    """Return a waterfall plot (dynamical spectrum)

    The input data has to be a two-dimensional array (it can be a
    masked array). The data will be flagged for bad channels,
    dedispersed and bandpass corrected from the (dedispersed)
    background.

    Note that the data is copied, so flagging and dedispersing does
    not alter the input data. This may be a an issue for large data
    sets.

    Parameters
    ----------

    For most of the arguments, see `core.create_dynspectrum`. Since
    the input data has be two-dimensional, options handling
    polarization data are not available.

    options: dict, optional
        The following options related to plotting are available

        - vmin, vmax: float. Set the range of the data to be included
          for the colormap (see also Matplotlib's vmin and vmax), in
          fractions. Default is vmin=0.1, vmax=0.9.

        - cmap: String. Color map to use. Default is "viridis".

        - cbar: Bool. Whether to draw a color bar on the side

        - fillmask: String. How to fill the masked values (flagged
          channels). Default is NaN, which tends to be
          background/transparanet values in Matplotlib. "mean" or
          "median" will replace masked values with the mean or median
          value of the non-masked data, respectively. You can also
          suply a function that takes the full data array as input
          (i.e., two-dimensional masked data) and returns a single
          value; so using `np.median` is the same as "median" in this
          case. Finally, you can give a single float or integer as
          replacement value.

        - xlabel, ylabel, x2label, y2label: Strings. The axis labels,
          for the x and y axes. The x2 and y2 labels are for the
          "derived" values: times and frequencies. Defaults are
          "samples", "channels", "time (milliseconds)" and "frequency
          (MHz)", respectively.

        - origin: String. Where to place the highest frequency. This
          relates to the 'origin' parameter from imshow. Default is
          "upper".

        - logscale: Bool. Whether to use apply a log scale to the
          input data. Default is False.

    """

    if not isinstance(data, (np.ndarray, np.ma.MaskedArray)):
        raise ValueError("data is not a NumPy (masked) array")
    if data.ndim != 2:
        raise ValueError("data is not two-dimensional")

    # Handle arguments
    if badchannels is None:
        badchannels = []

    vmin = options.get("vmin", 0.1)
    vmax = options.get("vmax", 0.9)
    cmap = options.get("cmap", "viridis")
    cbar = options.get("cbar", True)
    fillmask = options.get("fillmask", "nan")
    xlabel = options.get("xlabel", "samples")
    x2label = options.get("xlabel", "time (milliseconds)")
    ylabel = options.get("ylabel", "channels")
    y2label = options.get("ylabel", "frequency (MHz)")
    origin = options.get("origin", "upper")
    logscale = options.get("logscale", False)

    fig, ax = ensure_figure(ax)

    data = data.copy()  # Don't change the input data

    if badchannels is not None:
        data = core.flag(data, badchannels)

    stokesI, _ = core.create_dynspectrum(
        data, freqs, tsamp, dm, reffreq, backgroundrange, bkg_method, background
    )

    if fillmask:
        if isinstance(fillmask, (float, int)):
            stokesI = np.ma.filled(stokesI, fillmask)
        elif fillmask == "median":
            value = np.nanmedian(stokesI)
            stokesI = np.ma.filled(stokesI, value)
        elif fillmask == "mean":
            value = np.nanmedian(stokesI)
            stokesI = np.ma.filled(stokesI, value)
        elif callable(fillmask):
            value = fillmask(stokesI)
            stokesI = np.ma.filled(stokesI, value)
        else:  # default to NaN
            stokesI = np.ma.filled(stokesI, np.nan)

    if logscale:
        stokesI = symlog(stokesI)

    vmin, vmax = np.nanpercentile(stokesI, (vmin * 100, vmax * 100))

    image = ax.imshow(
        stokesI.T, aspect="auto", origin=origin, cmap=cmap, vmin=vmin, vmax=vmax
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if x2label:
        # Ensure things are in milliseconds
        dt = tsamp.to("ms").value
        axx2 = ax.secondary_xaxis("top", functions=(lambda x: x * dt, lambda x: x / dt))
        axx2.set_xlabel(x2label)
    if y2label:
        ax2 = ax.twinx()
        # Assume the frequencies are linear
        ax2.set_ylim([freqs[0].value, freqs[-1].value])
        ax2.set_ylabel(y2label)

    if cbar:
        if cbar == "left":
            fig.colorbar(image, ax=ax, orientation="vertical", location="left")
        else:
            fig.colorbar(image, ax=ax, orientation="vertical", pad=0.15)

    if return_image:
        return (fig, ax), image
    return (fig, ax)


def lightcurve(
    data: Array,
    freqs: np.ndarray,
    tsamp: float,
    dm: float | Quantity = 0,
    reffreq: float | None = None,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
    ax: Axes | None = None,
    **options,
) -> tuple[Figure, Axes]:
    """
    Create a light curve plot by summing across channels

    The data is corrected for dispersion and background first, taking into account any bad channels.

    """
    fig, ax = ensure_figure(ax)

    if badchannels is None:
        badchannels = []

    xlabel = options.get("xlabel", "samples")
    ylabel = options.get("ylabel", "intensity")
    logscale = options.get("logscale", False)
    ymin = options.get("ymin")

    data = data.copy()  # Don't change the input data

    if badchannels is not None:
        data = core.flag(data, badchannels)
    lightcurve, _ = core.calc_lightcurve(
        data, freqs, tsamp, dm, reffreq, backgroundrange, bkg_method=bkg_method
    )

    if logscale:
        lightcurve = symlog(lightcurve)
    if isinstance(ymin, (float, int)):
        lightcurve[lightcurve < ymin] = np.nan

    ax.plot(
        lightcurve,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return (fig, ax)


def background(
    data: Array,
    freqs: np.ndarray,
    tsamp: float,
    dm: float | Quantity = 0,
    reffreq: float | None = None,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    method: str = "mean",
    ax: Axes | None = None,
    **options,
) -> tuple[Figure, Axes]:
    """Create a background plot of the mean and std-dev of the background

    This plots the background, *after* dedispersion.

    Parameters:

    ax: None, or a list or tuple of two axis
        The first ax item is used for the mean background
        The second ax item is used for the std-dev background
    """
    fig, ax = ensure_figure(ax)

    data = data.copy()  # Don't change the input data

    if badchannels is not None:
        data = core.flag(data, badchannels)
    data = core.dedisperse(data, freqs, tsamp, dm, reffreq=reffreq)
    # `calc_background` by itself does not dedisperse or bandpass correct
    mean, stddev = core.calc_background(data, backgroundrange, method)

    if mean.ndim == 2:
        # Plot the background only for the first channel
        mean = mean[0]
        stddev = stddev[0]

    label_mean = options.get("label_mean", "mean bkg")
    label_std = options.get("label_std", "bkg stddev")
    xlabel = options.get("xlabel", "channels")
    ylabel = options.get("ylabel", "intensity")
    logscale = options.get("logscale", False)

    if logscale:
        mean = symlog(mean)
        stddev = symlog(stddev)

    channels = np.arange(1, len(freqs) + 1)
    ax.plot(channels, mean, label=label_mean)
    ax.plot(channels, stddev, label=label_std)
    ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return (fig, ax)


def bowtie(
    data: Array,
    freqs: np.ndarray,
    tsamp: float,
    dminterval: FInterval,
    reffreq: float | None = None,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
    ndm: int = 50,
    trange: slice | EllipsisType = Ellipsis,
    ax: Axes | None = None,
    **options,
) -> tuple[Figure, Axes]:
    """Create a bowtie plot: varying DM versus time/samples

    Parameters
    ----------


    dminterval : tuple[float, float] (FInterval)
        range of the dispersion measure: start and end

        A central DM is calculated from this range, and is the
        arithmetic mean of the start and end dispersion values.

    badchannels : set | list | np.ndarray | None, default=None
        numbers of channels to flag/ignore

    backgroundrange: 2-tuple of background interval, or iterable of
        2-tuples of background fraction intervals.

        Each interval is a 2-tuple that contains two floating point
        values between 0 and 1, which are the fractions of the full
        data sample (time-axis) range that contain a background
        section. All sections are combined, after which the background
        is calculated (using the median or mean value over the
        combined area).

        The background is calculcated once for the central DM (thus for
        ``(dm[0] + dm[1]) / 2``), so the datarange should be for the particular
        DM; all other DM samples use the same background value.

    ndm : int, default=50
        Number of DM samples along the y-axis

    reffreq: float or None

        reference frequency used for dispersion. If None, use the
        highest value of the given `freqs`.

    ax : Matplotlib Axes, default=None
        If given, use this axes to draw the graph on
    """
    # maxchan = len(burst.freqs)
    # badchannels = [maxchan - value for value in badchannels]

    fig, ax = ensure_figure(ax)

    data = data.copy()  # Don't change the input data

    if badchannels is not None:
        data = core.flag(data, badchannels)
    data, _ = core.bowtie(
        data,
        freqs,
        tsamp,
        dminterval,
        reffreq=reffreq,
        ndm=ndm,
        backgroundrange=backgroundrange,
        bkg_method=bkg_method,
    )

    # Calculate the extent for the imshow axes
    if isinstance(trange, EllipsisType):
        extent = [0, data.shape[1], dminterval[1], dminterval[0]]
    else:
        start = trange.start or 0
        stop = trange.stop if trange.stop else data.shape[1]
        extent = [start, stop, dminterval[1], dminterval[0]]

    # Ensure proper scalars
    if isinstance(extent[0], Quantity):
        extent[0] = extent[0].to("s").value
    if isinstance(extent[1], Quantity):
        extent[1] = extent[1].to("s").value
    if isinstance(extent[2], Quantity):
        extent[2] = extent[2].to(DMUNIT).value
    if isinstance(extent[3], Quantity):
        extent[3] = extent[3].to(DMUNIT).value

    vmin = options.get("vmin", 0.1)
    vmax = options.get("vmax", 0.9)
    cmap = options.get("cmap", "plasma")
    cbar = options.get("cbar", True)
    xlabel = options.get("xlabel", "samples")
    ylabel = options.get("ylabel", "DM")
    origin = options.get("origin", "lower")
    logscale = options.get("logscale", False)

    if logscale:
        data = symlog(data)

    vmin, vmax = np.nanpercentile(data, (vmin * 100, vmax * 100))

    image = ax.imshow(
        data,
        aspect="auto",
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    divider = make_axes_locatable(ax)
    if cbar is True or cbar.lower() == "right":
        cax = divider.append_axes("right", size="5%", pad=0.15)
    elif cbar.lower() == "left":
        cax = divider.append_axes("left", size="5%", pad=0.15)
    if cbar:
        ax.figure.colorbar(image, cax=cax, orientation="vertical")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def signal2noise(
    data: Array,
    freqs: np.ndarray,
    tsamp: float,
    dminterval: FInterval,
    reffreq: float | None = None,
    ndm: int = 50,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
    peak: bool = True,
    fit: bool = False,
    ax: Axes | None = None,
    **options,
) -> tuple[Figure, Axes]:
    fig, ax = ensure_figure(ax)

    data = data.copy()

    if badchannels is not None:
        data = core.flag(data, badchannels)
    dms, ratios = core.signal2noise(
        data,
        freqs,
        tsamp,
        dminterval=dminterval,
        reffreq=reffreq,
        ndm=ndm,
        backgroundrange=backgroundrange,
        bkg_method=bkg_method,
        peak=peak,
    )

    xlabel = options.get("xlabel", "DM")
    ylabel = options.get("ylabel", "S / N")
    logscale = options.get("logscale", False)

    if logscale:
        ratios = symlog(ratios)

    ax.plot(dms, ratios, "o")

    if fit:
        ampl, mean, stddev = core.fit_ratios(dms, ratios)
        x = np.linspace(dms[0], dms[-1])
        y = ampl * np.exp(-0.5 * (x - mean) ** 2 / stddev**2)
        ax.plot(x, y, "-")
        ax.hlines(
            [ampl, ampl - 1],
            0,
            1,
            transform=ax.get_yaxis_transform(),
            alpha=0.2,
            color="k",
            linestyle="--",
        )
        cuts = [
            mean - stddev * np.sqrt(-2 * np.log((ampl - 1) / ampl)),
            mean,
            mean + stddev * np.sqrt(-2 * np.log((ampl - 1) / ampl)),
        ]
        if cuts[0] < min(dms):
            cuts.pop(0)
        if cuts[-1] > max(dms):
            cuts.pop()
        ax.vlines(
            cuts,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            alpha=0.2,
            color="k",
            linestyle="--",
        )
        for cut in cuts:
            ax.text(
                cut,
                min(ratios),
                f"{cut:.5f}",
                ha="left",
                va="bottom",
                rotation="vertical",
            )
        if len(cuts) == 3:
            dcut = (cuts[2] - cuts[0]) / 2
            ax.text(
                0.98,
                0.98,
                rf"DM = {cuts[1]:.5f} $\pm$ {dcut:.5f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=14,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def grid(
    data: Array,
    freqs: np.ndarray,
    tsamp: float,
    dm: float,
    dminterval: FInterval,
    reffreq: float | None = None,
    ndm: int = 50,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
    peak: bool = True,
    peak_interval: FInterval | None = None,
    dm_coherent: float = 0,
    foff: float = 0,
    cfreq: float = 1,
    tstart: float | None = None,
    ax: Axes | None = None,
    **options,
) -> tuple[Figure, Axes]:
    """Plot multiple DM figures in one grid

    Parameters
    ----------

    dm : float

        Dispersion measure used to create the plots. For plots that
        require an interval, like the signal-to-noise plot,
        `dminterval` is used instead.

    dm_coherent : float, optional

        The coherent dm. Often the initial dispersion measure
        applied. Used to calculate the smearing; the `dm` argument is
        used for the actual plots. Default 0.

    """
    if not ax:
        figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    else:
        fig = ax.figure
        # Get the info from the original Axes
        subspec = ax.get_subplotspec()
        # Remove the original Axes
        fig.delaxes(ax)
        # Create a subfigure occupying the same region
        figure = fig.add_subfigure(subspec)

    title = options.get("title", "")
    gs = GridSpec(
        2,
        3,
        figure=figure,
        width_ratios=[0.1, 1, 0.5],
        height_ratios=[1, 2],
        wspace=0.05,
        hspace=0.05,
    )
    lc_ax = figure.add_subplot(gs[0, 1])
    info_ax = figure.add_subplot(gs[0, 2])
    w_ax = figure.add_subplot(gs[1, 1])
    c_ax = figure.add_subplot(gs[1, 0])
    s2n_ax = figure.add_subplot(gs[1, 2])
    _, image = waterfall(
        data,
        freqs,
        tsamp,
        dm,
        reffreq,
        badchannels=badchannels,
        backgroundrange=backgroundrange,
        bkg_method=bkg_method,
        return_image=True,
        ax=w_ax,
        cbar=False,
    )
    figure.colorbar(image, cax=c_ax, orientation="vertical")
    c_ax.yaxis.set_ticks_position("left")

    lightcurve(
        data,
        freqs,
        tsamp,
        dm,
        reffreq,
        badchannels=badchannels,
        backgroundrange=backgroundrange,
        bkg_method=bkg_method,
        ax=lc_ax,
    )
    lc_ax.set_title("Light curve")

    signal2noise(
        data,
        freqs,
        tsamp,
        dminterval,
        reffreq=reffreq,
        ndm=ndm,
        badchannels=badchannels,
        backgroundrange=backgroundrange,
        bkg_method=bkg_method,
        peak=peak,
        peak_interval=peak_interval,
        ax=s2n_ax,
    )
    s2n_ax.yaxis.set_label_position("right")
    s2n_ax.yaxis.tick_right()
    if peak:
        s2n_ax.set_title("Peak signal to noise")
    else:
        s2n_ax.set_title("Signal to noise")

    # Add overall info in top-right corner
    dm_incoherent = dm_coherent - dm
    smearing = abs(2 * DMCONST * dm_incoherent * foff * cfreq**-3)
    obsdate = (
        Time(tstart, format="mjd").strftime("%Y-%m-%dT%H:%M:%S.%f") if tstart else "-"
    )
    info_ax.axis("off")
    transform = info_ax.transAxes
    info_ax.text(
        0.0, 0.9, f"Burst: {title}", transform=transform, ha="left", fontsize=16
    )
    info_ax.text(0.0, 0.7, f"Obs-date: {obsdate}")
    info_ax.text(0.0, 0.6, f"DM: {dm:.3f}", transform=transform, ha="left")
    info_ax.text(
        0.0, 0.5, f"Coherent DM: {dm_coherent:.3f}", transform=transform, ha="left"
    )
    info_ax.text(0.0, 0.4, f"Smearing: {smearing:g}")
    now = Time.now().strftime("%Y-%m-%dT%H:%M:%S")
    info_ax.text(1.2, 1.1, f"Created {now}", ha="right", fontsize=9)

    return figure, w_ax
