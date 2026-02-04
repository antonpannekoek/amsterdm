"""Plotting functions for quick analysis plots

These functions provide basic plots for analysis of FRBs, such as a
waterfall plot (dynamical spectrum), a light curve, a "bowtie" plot, a
(peak) signal to noise graph, and an all-in-one plot

While some options can be set through keyword arguments, the functions
aim to provide only basic functionality; for publication-level
figures, one will likely want to create their own figures manually.

"""

from contextlib import suppress
from types import EllipsisType

from astropy.time import Time
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .candidate import Candidate
from .constants import DEFAULT_BACKGROUND_RANGE, DMCONST
from .utils import FInterval, symlog


def waterfall(
    candidate: Candidate,
    dm: float = 0,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    return_image: bool = False,
    ax=None,
    **options,
):
    """ """

    if not ax:
        figure = Figure(figsize=(12, 8))
        FigureCanvas(figure)
        ax = figure.add_subplot()

    if badchannels is None:
        badchannels = []

    vmin = options.get("vmin", 0.1)
    vmax = options.get("vmax", 0.9)
    cmap = options.get("cmap", "viridis")
    cbar = options.get("cbar", True)
    xlabel = options.get("xlabel", "samples")
    x2label = options.get("xlabel", "time (milliseconds)")
    ylabel = options.get("ylabel", "channels")
    y2label = options.get("ylabel", "frequency (MHz)")
    origin = options.get("origin", "upper")
    logscale = options.get("logscale", False)

    stokesI = candidate.create_dynspectrum(
        dm, badchannels, backgroundrange, bkg_method=bkg_method
    )

    stokesI = np.ma.filled(stokesI, np.nan)

    if logscale:
        stokesI = symlog(stokesI)

    vmin, vmax = np.nanpercentile(stokesI, (vmin * 100, vmax * 100))

    image = ax.imshow(
        stokesI.T, aspect="auto", origin=origin, cmap=cmap, vmin=vmin, vmax=vmax
    )

    divider = make_axes_locatable(ax)
    with suppress(AttributeError):
        cbar = cbar.lower()
    if cbar is True or cbar == "right":
        cax = divider.append_axes("right", size="5%", pad=0.15)
    elif cbar == "left":
        cax = divider.append_axes("left", size="5%", pad=0.15)
    if cbar:
        ax.figure.colorbar(image, cax=cax, orientation="vertical")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if x2label:
        # Ensure things are in milliseconds
        dt = candidate.header["tsamp"] * 1000
        axx2 = ax.secondary_xaxis("top", functions=(lambda x: x * dt, lambda x: x / dt))
        axx2.set_xlabel(x2label)
    if y2label:
        axy2 = ax.secondary_yaxis(
            "right", functions=(candidate.channel2freq, candidate.freq2channel)
        )
        axy2.set_ylabel(y2label)

    if return_image:
        return ax, image
    return ax


def lightcurve(
    candidate: Candidate,
    dm: float = 0,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    ax=None,
    **options,
):
    """
    Create a light curve plot by summing across channels

    The data is corrected for dispersion and background first, taking into account any bad channels.

    """

    if not ax:
        figure = Figure(figsize=(12, 8))
        FigureCanvas(figure)
        ax = figure.add_subplot()

    if badchannels is None:
        badchannels = []

    xlabel = options.get("xlabel", "samples")
    ylabel = options.get("ylabel", "intensity")
    logscale = options.get("logscale", False)
    ymin = options.get("ymin")

    # maxchan = len(candidate.freqs)
    # badchannels = [maxchan - value for value in badchannels]

    lightcurve = candidate.lightcurve(
        dm, badchannels, backgroundrange, bkg_method=bkg_method
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

    return ax


def background(
    candidate: Candidate,
    dm: float = 0,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    ax=None,
    **options,
):
    """Create a background plot of the mean and std-dev of the background

    Parameters:

    ax: None, or a list or tuple of two axis
        The first ax item is used for the mean background
        The second ax item is used for the std-dev background
    """

    _, bkg = candidate.create_dynspectrum(
        dm, badchannels, backgroundrange, bkg_method=bkg_method, bkg_extra=True
    )

    if not ax:
        figure = Figure(figsize=(12, 8))
        FigureCanvas(figure)
        # Create two axes, on top of each other
        ax = figure.add_subplot()

    label_mean = options.get("label_mean", "mean bkg")
    label_std = options.get("label_std", "bkg stddev")
    xlabel = options.get("xlabel", "channels")
    ylabel = options.get("ylabel", "intensity")
    logscale = options.get("logscale", False)

    mean, stddev = bkg["mean"], bkg["std"]
    if logscale:
        mean = symlog(mean)
        stddev = symlog(stddev)

    channels = np.arange(1, candidate.header["nchans"] + 1)
    ax.plot(channels, mean, label=label_mean)
    ax.plot(channels, stddev, label=label_std)
    ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def bowtie(
    candidate: Candidate,
    dm: FInterval,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    ndm: int = 50,
    reffreq: float | None = None,
    trange: slice | EllipsisType = Ellipsis,
    ax=None,
    **options,
):
    """Create a bowtie plot: varying DM versus time/samples

    Parameters
    ----------
    candidate : Candidate

    dm : tuple[float, float] (FInterval)
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

    # maxchan = len(candidate.freqs)
    # badchannels = [maxchan - value for value in badchannels]

    data = candidate.bowtie(
        dm,
        badchannels,
        backgroundrange,
        bkg_method,
        ndm,
        reffreq=reffreq,
    )

    # Calculate the extent for the imshow axes
    if isinstance(trange, EllipsisType):
        extent = [0, data.shape[1], dm[1], dm[0]]
    else:
        start = trange.start or 0
        stop = trange.stop if trange.stop else data.shape[1]
        extent = [start, stop, dm[1], dm[0]]

    if not ax:
        figure = Figure(figsize=(12, 8))
        FigureCanvas(figure)
        ax = figure.add_subplot()

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

    return ax


def signal2noise(
    candidate: Candidate,
    dms: np.ndarray,
    reffreq: float | None = None,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    peak: bool = True,
    peak_interval: FInterval | None = None,
    ax=None,
    **options,
):
    ratios = candidate.signal2noise(
        dms=dms,
        reffreq=reffreq,
        badchannels=badchannels,
        backgroundrange=backgroundrange,
        bkg_method=bkg_method,
        peak=peak,
        peak_interval=peak_interval,
    )
    if not ax:
        figure = Figure(figsize=(12, 8))
        FigureCanvas(figure)
        ax = figure.add_subplot()

    xlabel = options.get("xlabel", "DM")
    ylabel = options.get("ylabel", "S / N")
    logscale = options.get("logscale", False)

    if logscale:
        ratios = symlog(ratios)

    ax.plot(dms, ratios, "o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def grid(
    candidate: Candidate,
    dm: float,
    dms: np.ndarray,
    reffreq: float | None = None,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    peak: bool = True,
    peak_interval: FInterval | None = None,
    coherent_dm: float = 0,
    ax=None,
    **options,
):
    if not ax:
        figure = Figure(figsize=(12, 8), constrained_layout=True)
        FigureCanvas(figure)
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
        candidate,
        dm=dm,
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
        candidate,
        dm=dm,
        badchannels=badchannels,
        backgroundrange=backgroundrange,
        bkg_method=bkg_method,
        ax=lc_ax,
    )
    lc_ax.set_title("Light curve")

    signal2noise(
        candidate,
        dms,
        reffreq=reffreq,
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
    incoherent_dm = coherent_dm - dm
    smearing = abs(
        2 * DMCONST * incoherent_dm * candidate.header["foff"] * candidate.cfreq**-3
    )
    obsdate = candidate.header.get("tstart")
    obsdate = (
        Time(obsdate, format="mjd").strftime("%Y-%m-%dT%H:%M:%S.%f") if obsdate else "-"
    )
    # smearing = 2 * DMCONST * incoherent_dm * candidate.channel_bandwidth_mhz * central_frequency_mhz ** -3
    info_ax.axis("off")
    transform = info_ax.transAxes
    info_ax.text(
        0.0, 0.9, f"Burst: {title}", transform=transform, ha="left", fontsize=16
    )
    info_ax.text(0.0, 0.7, f"Obs-date: {obsdate}")
    info_ax.text(0.0, 0.6, f"DM: {dm:.3f}", transform=transform, ha="left")
    info_ax.text(
        0.0, 0.5, f"Coherent DM: {coherent_dm:.3f}", transform=transform, ha="left"
    )
    info_ax.text(0.0, 0.4, f"Smearing: {smearing:g}")
    now = Time.now().strftime("%Y-%m-%dT%H:%M:%S")
    info_ax.text(1.2, 1.1, f"Created {now}", ha="right", fontsize=9)

    return w_ax
