import numpy as np

from .constants import DEFAULT_BACKGROUND_RANGE, DMCONST
from .utils import FInterval


__all__ = [
    "calc_background",
    "correct_bandpass",
    "dedisperse",
    "create_dynspectrum",
    "bowtie",
    "signal2noise",
]


def downsample(
    data: np.ndarray | np.ma.MaskedArray,
    factor: int = 8,
    remainder: str = "droptail",
    method: str = "mean",
) -> np.ndarray | np.ma.MaskedArray:
    """Downsample `data` by `factor` along the first axis. Bins can be
    averaged (default) or summed together.

    If the first axis doesn't match an integer number of `factor`, the
    remainder can be dropped, either from the start ("drophead") or
    the end ("droptail"; the default); or the remainder can be added
    to the last bin ("addtail") or be added to the first bin
    ("addhead").

    If the number of available bins in `data` is smaller than
    `factor`, all bins are combined, even when `method` is one of
    "droptail" or "drophead".

    Raises a `ValueError`
        - for incorrect data dimensions (less than 2)
        - for an incorrect `factor` (less than 1)
        - for an incorrect remainder value
        - for an incorrect method

    """

    if data.ndim < 2:
        raise ValueError("'data' should at least be two-dimensional")
    if factor < 1:
        raise ValueError("'factor' should be a positive integer")
    if method not in ("mean", "sum"):
        raise ValueError("'method' should be one of 'mean' or 'sum'")
    if remainder not in ("droptail", "addtail", "drophead", "addhead"):
        raise ValueError(
            "'remainder' should be one of 'droptail', 'addtail', 'drophead' or 'addhead'"
        )

    n = data.shape[0]
    if n <= factor:
        if method == "mean":
            return data.mean(axis=0, keepdims=True)
        elif method == "sum":
            return data.sum(axis=0, keepdims=True)

    nbins, rem = divmod(n, factor)
    combbin = factor + rem
    if "tail" in remainder:
        indices = np.arange(0, nbins * factor, factor)
        count = factor * np.ones(nbins)
        if "add" in remainder:
            count[-1] = combbin
        elif "drop" in remainder and rem > 0:
            data = data[:-rem, ...]
            # if rem > 0:
            #    s = slice(None, -rem)
    elif "addhead" in remainder:
        indices = np.hstack([[0], combbin + np.arange(0, nbins - 1) * factor])
        count = factor * np.ones(nbins)
        count[0] = combbin
    elif "drophead" in remainder:
        indices = rem + np.arange(0, n - rem, factor)
        count = factor * np.ones(nbins)

    summed = np.add.reduceat(data, indices, axis=0)

    shape = (nbins,) + (1,) * (data.ndim - 1)
    count = count.reshape(shape)

    if method == "sum":
        return summed
    else:
        return summed / count


def calc_background_old(
    data: np.ndarray | np.ma.MaskedArray,
    datarange: tuple[float, float] = (0.3, 0.7),
    method: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return background and its standard deviation for each channel

    Assumes any dispersion correction has already been done

    For each channel, a background level is estimated. This is done by
    selecting a set of time-sample columns outside of the columns that contain
    actual object of interest (the latter is given with ``datarange`` as a
    range in fraction of the total time-sample columns), then calculating an
    average, using one of three ``method``s (mean, median or mode), and the
    background standard deviation (noise). The average is subtracted from the
    full channel values, then the channel is divided (normalized) by the
    standard deviation.

    Parameters
    ----------
    data : np.ndarray | np.ma.MaskedArray
        data that needs be normalised. Usually contains frequency on the y-axis
        and time samples on the x-axis.

    datarange : tuple[float, float], default=(0.3, 0.7)

        Fractional range along the time axis, where the actual object is
        located. Data outside these columns is used as the background for the
        estimation of mean/median/mode and standard deviation for the bandpass
        correction.

    method : str, default="median"
        method to estimate the background level for each channel.

        Note that "mode" is not very applicable for continuously distributed
        data; and for normally distributed data, it will be the same value as
        the median or mean.
    """

    if method not in ["mean", "median", "mode"]:
        raise ValueError("method should be one of 'mean', 'median' or 'mode'")

    nsamp = data.shape[0]
    bkgrange = (int(nsamp * datarange[0]), int(nsamp * datarange[1]))

    idx = np.arange(nsamp)
    idx_bg = [np.array([], dtype=int)]
    if bkgrange[0] > 0:
        idx_bg.append(idx[: bkgrange[0]])
    if bkgrange[1] < nsamp:
        idx_bg.append(idx[bkgrange[1] :])

    idx_bg = np.concatenate(idx_bg)
    # bkg = np.ma.filled(data[idx_bg, :], np.nan)
    bkg = data[idx_bg, :]

    if method == "mean":
        mean = np.ma.mean(bkg, axis=0)
    elif method == "median":
        mean = np.ma.median(bkg, axis=0)
    elif method == "mode":
        mean = np.empty(data.shape[1])
        for i in range(data.shape[1]):
            hist, bin_edges = np.histogram(bkg[:, i], bins=100)
            max_bin = np.argmax(hist)
            mean[i] = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])
    else:  # we shouldn't be able to get here
        raise ValueError("method should be one of 'mean', 'median' or 'mode'")

    std = np.ma.std(bkg, axis=0)

    return mean, std


def calc_background(
    data: np.ndarray | np.ma.MaskedArray,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    method: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return background and its standard deviation for each channel

    Assumes any dispersion correction has already been done

    For each channel, a background level is estimated. This is done by
    selecting a set of time-sample columns outside of the columns that contain
    actual object of interest (the latter is given with ``datarange`` as a
    range in fraction of the total time-sample columns), then calculating an
    average, using one of three ``method``s (mean, median or mode), and the
    background standard deviation (noise). The average is subtracted from the
    full channel values, then the channel is divided (normalized) by the
    standard deviation.

    Parameters
    ----------
    data : np.ndarray | np.ma.MaskedArray
        data that needs be normalised. Usually contains frequency on the y-axis
        and time samples on the x-axis.

    backgroundrange: iterable of 2-tuples
        Iterable of ranges as fractions of the sample dimension of the data, that is,
        each iterable item contains a start and end fraction of the first dimension of
        the data that corresponds to a background area

    method : str, default="median"
        method to estimate the background level for each channel.

        Note that "mode" is not very applicable for continuously distributed
        data; and for normally distributed data, it will be the same value as
        the median or mean.
    """

    if method not in ["mean", "median", "mode"]:
        raise ValueError("method should be one of 'mean', 'median' or 'mode'")

    if isinstance(backgroundrange[0], (float, int)):
        backgroundrange = [backgroundrange]

    nsamp = data.shape[0]
    idx_bkg = []
    for bkgrange in backgroundrange:
        low = int(nsamp * bkgrange[0] + 0.5)
        high = int(nsamp * bkgrange[1] + 0.5)
        idx_bkg.append(np.arange(low, high))
    idx_bkg = np.concatenate(idx_bkg)
    bkg = data[idx_bkg, :]

    if method == "mean":
        mean = np.ma.mean(bkg, axis=0)
    elif method == "median":
        mean = np.ma.median(bkg, axis=0)
    elif method == "mode":
        mean = np.empty(data.shape[1])
        for i in range(data.shape[1]):
            hist, bin_edges = np.histogram(bkg[:, i], bins=100)
            max_bin = np.argmax(hist)
            mean[i] = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])
    else:  # we shouldn't be able to get here
        raise ValueError("method should be one of 'mean', 'median' or 'mode'")

    std = np.ma.std(bkg, axis=0)

    return mean, std


def correct_bandpass(
    data: np.ndarray | np.ma.MaskedArray,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    method: str = "median",
    extra: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correct for the individual channel bandpasses in a given data array

    Perform a bandpass correction: scale each channel with its background to
    correct for different sensitivities per channel.

    For each channel, a background level is estimated. This is done by
    selecting a set of time-sample columns outside of the columns that contain
    actual object of interest (the latter is given with ``datarange`` as a
    range in fraction of the total time-sample columns), then calculating an
    average, using one of three ``method``s (mean, median or mode), and the
    background standard deviation (noise). The average is subtracted from the
    full channel values, then the channel is divided (normalized) by the
    standard deviation.

    Parameters
    ----------
    data : np.ndarray | np.ma.MaskedArray
        data that needs be normalised. Usually contains frequency on the y-axis
        and time samples on the x-axis.

    backgroundrange: 2-tuple of background interval, or iterable of
        2-tuples of background fraction intervals.

        Each interval is a 2-tuple that contains two floating point
        values between 0 and 1, which are the fractions of the full
        data sample (time-axis) range that contain a background
        section. All sections are combined, after which the background
        is calculated (using the median or mean value over the
        combined area).

    method : str, default="median"
        method to estimate the background level for each channel.

        Note that "mode" is not very applicable for continuously distributed
        data; and for normally distributed data, it will be the same value as
        the median or mean.

    extra : bool, default=False
        if True, return the bandpass-corrected data, the background averages
        and the background standard devations. If False, return only the
        bandpass-corrected data.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    mean, std = calc_background(data, backgroundrange, method)

    # Bandpass correction
    data_sub = (data - mean[None, :]) / std[None, :]

    if extra:
        return data_sub, mean, std

    return data_sub


def dedisperse(
    data: np.ndarray | np.ma.MaskedArray,
    dm: float,
    freqs: np.ndarray,
    tsamp: float,
    reffreq: float | None = None,
    dmconst: float = DMCONST,
) -> np.ndarray:
    """
    Dedisperse a two-dimensional data set

    Parameters
    ----------
    data : np.ndarray | np.ma.MaskedArray
        data containing freq on the y-axis and time on the x-axis
    dm : float
        Dispersion measure in units of pc / cc.
    freqs : np.ndarray
        Array containing the channel frequencies in units of MHz
    tsamp : float
        sampling time in units of seconds.
    dmconst : float, default=DMCONST
        dispersion constant in units of MHz^2 pc^-1 cm^3 s

    Returns
    -------
    ndata : np.ndarray
        dedispersed data
    """
    # set up freq info
    freqs = freqs.astype(np.float64)
    if reffreq is None:
        reffreq = np.max(freqs)

    # init empty array to store dedisp data in
    newdata = np.empty_like(data)
    # `empty_like` copies the mask from `data`
    np.testing.assert_equal(newdata.mask, data.mask)

    # calculate time shifts and convert to bin shifts
    time_shift = dmconst * dm * (reffreq**-2.0 - freqs**-2.0)

    # round to nearest integer
    bin_shift = np.rint((time_shift / tsamp)).astype(np.int64)

    # checks
    assert len(bin_shift) == data.shape[0]

    # dedisperse by rolling back the channels
    for i, shift in enumerate(bin_shift):
        newdata[i, :] = np.roll(data[i, :], shift)

    return newdata


def calc_intensity(
    data: dict[str, np.ndarray],
    dm: dict[str, float | np.ndarray] | None = None,
    badchannels: set | list | np.ndarray | None = None,
    datarange: tuple[float, float] | None = (0.3, 0.7),
    bkg_method: str = "mean",
    bkg_extra: bool = False,
):
    """
    Returns the Stokes I / intensity parameter from the xx and yy data

    It will optionally correct for bad channels, bandpass and dispersion, if
    the relevant keyword argument is given.

    Parameters
    ----------

    data : dict[str, np.ndarray]
        array for value.

    dm : dict[str, float | np.ndarray] | None, default=None

         The ``dm`` dict should contain the disperson measure "dm", the
         frequencies corresponding to the channels "freq" and the timestamps
         corresponding to the time-samples "tsamp".

         Dedisperse the data for the given value. The default value of None
         means no dedispersion is applied.

    badchannels : set | list | np.ndarray | None, default=None
        means no flagging is done.

    datarange : tuple[float, float] | None, default=(0.3, 0.7)

        Fractional range along the time axis, where the actual object is
        located. Data outside these columns is used for the bandpass
        correction.

        The default of None indicates no bandpass correction is applied.

    bkg_extra : bool, default=False

        If ``True``, returns an additional object, which is a dict containing
        the mean and standard deviation of the background along the channels;
        these are one-dimensional arrays

    Returns
    -------

        Two-dimensional array with the Stokes intensity parameter. If
        ``bkg_extra`` is ``True``, returns a two-tuple of (two-dimensional
        array, bkg_info dict).
    """

    xx = np.ma.array(data["xx"])
    yy = np.ma.array(data["yy"])

    if badchannels:
        rowids = (
            list(badchannels)
            if not isinstance(badchannels, (list, np.ndarray))
            else badchannels
        )
        xx[:, rowids] = np.ma.masked
        yy[:, rowids] = np.ma.masked

    if dm:
        xx = dedisperse(xx.T, dm["dm"], dm["freq"], dm["tsamp"]).T
        yy = dedisperse(yy.T, dm["dm"], dm["freq"], dm["tsamp"]).T

    extra = {}
    if datarange:
        xx_bkgmean, xx_bkgstd = calc_background(xx, datarange)
        yy_bkgmean, yy_bkgstd = calc_background(yy, datarange)

        # Bandpass correction
        xx = (xx - xx_bkgmean[None, :]) / xx_bkgstd[None, :]
        yy = (yy - yy_bkgmean[None, :]) / yy_bkgstd[None, :]

        if bkg_extra:
            extra["mean"] = xx_bkgmean + yy_bkgmean
            extra["std"] = np.sqrt(xx_bkgstd**2 + yy_bkgstd**2)

    intensity = xx + yy

    if extra:
        return intensity, extra

    return intensity


def create_dynspectrum(
    data: np.ndarray,
    dm: dict[str, float | np.ndarray] | None = None,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    bkg_extra: bool = False,
    background: tuple[float | dict, float | dict] | None = None,
):
    """Returns a dynamical spectrum with the Stokes I / intensity
    parameter from the input data data array

    The routine flags bad channels, corrects for the given dispersion,
    calculates a background and corrects for the bandpass.

    When multiple polarization channels (xx and yy) exist, it does
    this for each independently, then combines the values together.

    The data is either

    - two-dimensional, with the first dimension the time samples and
      the second dimension the frequency channels,

    - or three-dimensional, with the first dimension the time samples,
      the second dimension the polarization, and the third dimension
      the frequency channels.

    The data should contain either one or no polarization dimension, in
    which case this is assumed to be Stokes I; or the data contains four
    polarization channels, of which the first two are assumed to be xx
    and yy.

    It will optionally correct for bad channels, bandpass and dispersion, if
    the relevant keyword argument is given.

    Parameters
    ----------

    data : dict[str, np.ndarray]
        array for value.

    dm : dict[str, float | np.ndarray] | None, default=None

         The ``dm`` dict should contain the disperson measure "dm", the
         frequencies corresponding to the channels "freq" and the timestamps
         corresponding to the time-samples "tsamp".

         Dedisperse the data for the given value. The default value of None
         means no dedispersion is applied.

    badchannels : set | list | np.ndarray | None, default=None
        means no flagging is done.

        The bad channels are assumed to be the same for the xx and yy
        polarizations, if applicable.

    backgroundrange: 2-tuple of background interval, or iterable of
        2-tuples of background fraction intervals.

        Each interval is a 2-tuple that contains two floating point
        values between 0 and 1, which are the fractions of the full
        data sample (time-axis) range that contain a background
        section. All sections are combined, after which the background
        is calculated (using the median or mean value over the
        combined area).

    bkg_extra : bool, default=False

        If ``True``, returns an additional object, which is a dict containing
        the mean and standard deviation of the background along the channels;
        these are one-dimensional arrays

    background: tuple of mean and standard deviation of the background values

        The tuple values can also be dicts. In that case, the keys are
        the polarization keys, (xx and yy), with the valuse the mean
        and standard deviation for those polarization parts. If the
        tuple elements are single values, but the input data contains
        multiple polarizations, it is assumed that the mean and
        standard deviation are the same for xx and yy.


    If the `background` argument is given, `backgroundrange` and
    `bkg_method` are ignored. If `bkg_extra` is also set, the returned
    values identical to the given values.

    Returns
    -------

        Two-dimensional array with the Stokes intensity parameter. If
        ``bkg_extra`` is ``True``, returns a two-tuple of (two-dimensional
        array, bkg_info dict).

    """

    data = np.squeeze(data)
    if data.ndim == 2:
        xx = np.ma.array(data)
        yy = None
    elif data.ndim != 3:
        raise ValueError("data has incorrect number of dimensions")
    elif data.shape[1] != 4:
        raise ValueError("second (polarization) dimension has incorrect size")
    else:
        xx = np.ma.array(data[:, 0, :])
        yy = np.ma.array(data[:, 1, :])

    if badchannels:
        rowids = (
            list(badchannels)
            if not isinstance(badchannels, (list, np.ndarray))
            else badchannels
        )
        xx[:, rowids] = np.ma.masked
        if yy is not None:
            yy[:, rowids] = np.ma.masked

    if dm:
        xx = dedisperse(xx.T, dm["dm"], dm["freq"], dm["tsamp"]).T
        if yy is not None:
            yy = dedisperse(yy.T, dm["dm"], dm["freq"], dm["tsamp"]).T

    if background:
        # Use a given background
        bkgmean, bkgstd = background
        if isinstance(bkgmean, dict):
            # Separate background values for xx and yy
            xx_bkgmean = bkgmean["xx"]
            yy_bkgmean = bkgmean["yy"]
        else:
            xx_bkgmean = yy_bkgmean = bkgmean
        if isinstance(bkgstd, dict):
            # Separate background values for xx and yy
            xx_bkgstd = bkgstd["xx"]
            yy_bkgstd = bkgstd["yy"]
        else:
            xx_bkgstd = yy_bkgstd = bkgstd
    else:
        # Calculate the background from the data
        xx_bkgmean, xx_bkgstd = calc_background(xx, backgroundrange, method=bkg_method)
        yy
        if yy is not None:
            yy_bkgmean, yy_bkgstd = calc_background(
                yy, backgroundrange, method=bkg_method
            )

    # Perform the bandpass correction using the background
    xx = (xx - xx_bkgmean[None, :]) / xx_bkgstd[None, :]
    if yy is not None:
        yy = (yy - yy_bkgmean[None, :]) / yy_bkgstd[None, :]
        # Add the two polarization channels together
        intensity = xx + yy
    else:
        intensity = xx
        yy_bkgmean = yy_bkgstd = 0

    if bkg_extra:
        extra = {
            "mean": xx_bkgmean + yy_bkgmean,
            "std": np.sqrt(xx_bkgstd**2 + yy_bkgstd**2),
        }

    if bkg_extra:
        return intensity, extra

    return intensity


def calc_lightcurve(
    data: dict[str, np.ndarray],
    dm: dict[str, float | np.ndarray] | None = None,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    bkg_extra: bool = False,
):
    """Calculate the light curve by summing across channels, after
    dedispersion, flagging bad channels and background correction.

    This returns a one-dimensional array of summed intensity versus
    samples. Optionally, the average standard deviation is returned.

    The light curve is computed from the two-dimensional intensity
    array, and the arguments are identical to that of
    ``calc_intensity``.

    Parameters
    ----------

    data : dict[str, np.ndarray]
        array for value.

    dm : dict[str, float | np.ndarray] | None, default=None

         The ``dm`` dict should contain the disperson measure "dm", the
         frequencies corresponding to the channels "freq" and the timestamps
         corresponding to the time-samples "tsamp".

         Dedisperse the data for the given value. The default value of None
         means no dedispersion is applied.

    badchannels : set | list | np.ndarray | None, default=None
        means no flagging is done.

    backgroundrange: 2-tuple of background interval, or iterable of
        2-tuples of background fraction intervals.

        Each interval is a 2-tuple that contains two floating point
        values between 0 and 1, which are the fractions of the full
        data sample (time-axis) range that contain a background
        section. All sections are combined, after which the background
        is calculated (using the median or mean value over the
        combined area).

    bkg_extra : bool, default=False

        If ``True``, returns an additional object, which is the
        average standard deviation, calculated from the background
        area for full two-dimensional data (see also
        ``calc_background``).

    Returns
    -------

        A light curve in the form of a one-dimensional array with of intensities versus samples.
        If ``bkg_extra`` is ``True``, returns a two-tuple of (light curve, average standard deviation).

    """

    results = create_dynspectrum(
        data,
        dm,
        badchannels,
        backgroundrange,
        bkg_method=bkg_method,
        bkg_extra=bkg_extra,
    )

    if bkg_extra:
        results, bkg = results
        _, std = bkg
        # the background for the intensity is only summed across
        # samples, for each channel separately thus we need to average
        # this array of standard deviations
        std = np.sqrt(np.mean(std**2))

    lightcurve = results.sum(axis=1)

    return (lightcurve, std) if bkg_extra else lightcurve


def calc_lightcurve_from_waterfall(waterfall, bkg_extra=None):
    """Calculate the light curve from waterfall data

    If ``bkg_extra`` is provided with the extra background data from
    the waterfall calculation, an average standard deviation will be
    returned as well.

    """

    if bkg_extra is not None:
        _, std = bkg_extra
        # the background for the intensity is only summed across
        # samples, for each channel separately thus we need to average
        # this array of standard deviations
        std = np.sqrt(np.mean(std**2))

    lightcurve = waterfall.sum(axis=1)

    return lightcurve if bkg_extra is None else (lightcurve, std)


def bowtie(
    data: np.ndarray,
    dm: FInterval,
    freqs: np.ndarray,
    tsamp: float,
    badchannels: set | list | np.ndarray | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    ndm: int = 50,
) -> np.ndarray:
    """Create the data for a bowtie plot: varying DM versus time/samples

    Parameters
    ----------
    data : np.ndarray
        data containing freq on the y-axis and time on the x-axis

    dm : tuple[float, float]
        range of the dispersion measure: start and stop

        An average DM is calculated from this range, which is then
        used in the calculation of the background: the data is
        dedispersed to this mean DM and the background is calculated,
        which is used for the bandpass correction.

    freqs : np.ndarray
        frequencies corresponding to the channel centers

    tsamp : float
        sampling time interval in seconds

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

        The background is determined with respect to the average DM of
        the given `dm` interval.

    ndm : int, default=50
        Number of DM samples along the y-axis

    Returns
    -------
    np.ndarray

    """

    dm_center = (dm[0] + dm[1]) / 2

    # Copy the data, flag bad channels and get a background estimate for the central DM
    # After that, perform a bandpass correction and dedisperse to the central DM
    # This provides the starting point for varying DM
    # This assumes linear XX and YY polarization channels

    data = np.squeeze(data)
    if data.ndim == 2:
        xx = np.ma.array(data)
        yy = None
    elif data.ndim != 3:
        raise ValueError("data has incorrect number of dimensions")
    elif data.shape[1] != 4:
        raise ValueError("second (polarization) dimension has incorrect size")
    else:
        xx = np.ma.array(data[:, 0, :])
        yy = np.ma.array(data[:, 1, :])

    if badchannels:
        rowids = (
            list(badchannels)
            if not isinstance(badchannels, (list, np.ndarray))
            else badchannels
        )
        xx[:, rowids] = np.ma.masked
        if yy is not None:
            yy[:, rowids] = np.ma.masked

    xx = dedisperse(xx.T, dm_center, freqs, tsamp).T
    if yy is not None:
        yy = dedisperse(yy.T, dm_center, freqs, tsamp).T

    # Bandpass correction
    mean, std = calc_background(xx, backgroundrange, method=bkg_method)
    xx = (xx - mean[None, :]) / std[None, :]
    if yy is not None:
        mean, std = calc_background(yy, backgroundrange, method=bkg_method)
        yy = (yy - mean[None, :]) / std[None, :]

    # xx and yy are now flagged, bandpass-corrected and dedispersed at the mean DM
    # This provides the starting point for the iteration through dmrange

    # dmrange is relative to the mean DM
    dmrange = np.linspace(dm[0], dm[1], ndm) - dm_center

    tie = []
    for dm in dmrange:
        xxdd = dedisperse(xx.T, dm, freqs, tsamp).T
        if yy:
            yydd = dedisperse(yy.T, dm, freqs, tsamp).T
            stokesI = xxdd + yydd
        else:
            stokesI = xxdd
        stokesI = np.ma.filled(stokesI, np.nan)
        lc = np.nansum(stokesI, axis=1)
        tie.append(lc)
    tie = np.vstack(tie)

    return tie
