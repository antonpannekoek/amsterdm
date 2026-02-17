import logging

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


logger = logging.getLogger(__package__)


def downsample(
    data: np.ndarray | np.ma.MaskedArray,
    factor: int = 1,
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


def upsample(
    data: np.ndarray,
    factor: int = 1,
) -> np.ndarray:
    """Rebin the data to a higher resolution along the first (sample/time) axis

    Sample bins are simply split into `factor` new bins, with the same
    value as that of the original bin.

    Under the hood, this simply uses `numpy.repeat` for the first
    axis.

    """

    if factor < 1:
        raise ValueError("'factor' should be a positive integer")

    return np.repeat(data, factor, axis=0)


def findpeaklc(
    data: np.ndarray | np.ma.MaskedArray,
    searchrange: tuple[float, float] = (0, 1),
) -> int:
    """Find the peak of the light curve, within an optional search range

    data: the one-dimensional light curve intensity data

    searchrange: a 2-tuple of floats
        Fractional start and end of the search range

    This simply returns the index of the maximum of `data`, potentially
    restricted to within a fraction of the data by `searchrange`.

    """

    n = len(data)
    low, high = int(searchrange[0] * n + 0.5), int(searchrange[1] * n + 0.5)
    index = np.argmax(data[low:high]) + low
    return index


def findrangelc(
    data: np.ndarray | np.ma.MaskedArray,
    kappa: float = 10,
    minkappa: float = 3,
    window: int = 7,
    maxiter: int = 10,
    minvalues: int = 10,
    searchrange: tuple[float, float] = (0, 1),
    bkg_extra: bool = False,
) -> tuple[int, int] | tuple[tuple[int, int], tuple[float, float]]:
    """Find the range of the active light curve.

    data: the one-dimensional light curve intensity data

    kappa: float
        find peaks `kappa` times the noise above

    minkappa: minimal noise value to be included

    window: integer
        number of bins to use in the rolling average.

    searchrange: a 2-tuple of floats
        Fractional start and end of the search range

    The algorithm first smooths the light curve by using a rolling
    average with `window` size.

    It then iteratively (up to `maxiter` times)
      - calculates a mean value
      - find all values below that mean
      - removes all non-found values
    Iteration stops when there are less than `minvalues` (default of
    10) values left.

    The remaining data are seen as the background. It takes the
    indices of the remaining data, and calculates a median and
    standard deviation from the non-smoothed data for this selection
    of indices; this is used as a first estimate for the background
    value and its noise.

    It then finds all values in the smoothed data that are `kappa`
    times noise above the background. The relevant indices are
    combined into sections, and each of these sections are extended on
    both sides to a `minkappa` times the noise above the
    background. The latter step is done separately, so that incidental
    low-sigma spikes above the background are not included, only when
    adjacent to a larger foreground region.

    These sections then define the foreground area where there is an
    active light curve.

    The sections are then returned, as a list of 2-tuples with start
    and end indices.

    Returns
       A tuple of 2 items:
       - A list of 2-tuples of integers. These represent the start and end indices
         of sections where the light curve is active.
       - Optionally, a tuple of the estimated background value and standard deviation

    """

    # Smooth the data with a window
    sdata = np.convolve(data, np.ones(window), mode="same") / window

    selection = np.ones(len(sdata), dtype=bool)
    for i in range(maxiter):
        mean = sdata[selection].mean()
        selection = selection & (sdata < mean)
        if selection.sum() < minvalues:
            break
    bkgval = np.ma.median(data[selection])
    bkgstd = data[selection].std()

    # With the background determined from the full data, limit the
    # search area
    n = len(data)
    low, high = int(searchrange[0] * n + 0.5), int(searchrange[1] * n + 0.5)
    sdata = sdata[low:high]
    data = data[low:high]

    above = sdata > bkgval + kappa * bkgstd

    indices = np.where(np.diff(above))[0]

    if above[0]:  # first section starts above the background
        indices = np.hstack([[0], indices])
    # Append a closing index if there is an open section at the end
    if len(indices) % 2 == 1:
        indices = np.append(indices, [n - 1])

    # Indices containing everything below the kappa-sigma background
    bkgindices = np.where(sdata <= (bkgval + minkappa * bkgstd))[0]

    # Create the sections pairs
    sections = []
    for index1, index2 in zip(indices[::2], indices[1::2]):
        # Find the first index to the left of index1 that is above the background
        sel = bkgindices < index1
        if sel.any():
            index = bkgindices[sel][-1] + 1
            if index < index1:
                index1 = index
        # Find the first index to the right of index2 that is above the background
        sel = bkgindices > index2
        if sel.any():
            index = bkgindices[sel][0] - 1
            if index > index2:
                index2 = index
        sections.append([index1, index2])
    # Combine overlapping sections
    remove = []
    for i, (section1, section2) in enumerate(zip(sections[:-1], sections[1:])):
        if section1[1] >= section2[0]:
            # Extend section2
            section2[0] = section1[0]
            # and remove section1
            remove.append(i)
    for i in reversed(remove):
        sections.pop(i)
    sections = [(section[0] + low, min(section[1] + low, high)) for section in sections]

    if bkg_extra:
        return sections, (bkgval, bkgstd)
    return sections


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
        each iterable item contains a begin and end fraction of the first dimension of
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

    # calculate time shifts and convert to bin shifts
    time_shift = dmconst * dm * (reffreq**-2.0 - freqs**-2.0)
    # round to nearest integer
    bin_shift = np.rint((time_shift / tsamp)).astype(np.int64)
    # check
    assert len(bin_shift) == data.shape[0]

    # init empty array to store dedisp data in
    newdata = np.empty_like(data)
    # `empty_like` copies the mask from `data`
    np.testing.assert_equal(newdata.mask, data.mask)

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
    if badchannels is not None:
        rowids = (
            list(badchannels)
            if not isinstance(badchannels, (list, np.ndarray))
            else badchannels
        )

        xx[:, rowids] = np.ma.masked
        if yy is not None:
            yy[:, rowids] = np.ma.masked

    if dm:
        reffreq = dm.get("reffreq")
        xx = dedisperse(xx.T, dm["dm"], dm["freq"], dm["tsamp"], reffreq=reffreq).T
        if yy is not None:
            yy = dedisperse(yy.T, dm["dm"], dm["freq"], dm["tsamp"], reffreq=reffreq).T

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
    # Note: since xx is a masked array, any division by zero
    # (in the stddev data) will result in those entries being masked
    # This is convenient `np.ma` behaviour.See
    # https://numpy.org/doc/stable/reference/maskedarray.generic.html#operations-on-masked-arrays
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
    reffreq: float | None = None,
) -> np.ndarray:
    """Create the data for a bowtie plot: varying DM versus time/samples

    Arguments
    ---------
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

    reffreq: float or None

        reference frequency used for dispersion. If None, use the
        highest value of the given `freqs`.

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
    two dimensional array containing the bowtie-plot data

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

    if badchannels is not None:
        rowids = (
            list(badchannels)
            if not isinstance(badchannels, (list, np.ndarray))
            else badchannels
        )
        xx[:, rowids] = np.ma.masked
        if yy is not None:
            yy[:, rowids] = np.ma.masked

    xx = dedisperse(xx.T, dm_center, freqs, tsamp, reffreq).T
    if yy is not None:
        yy = dedisperse(yy.T, dm_center, freqs, tsamp, reffreq).T

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
        if yy is not None:
            yydd = dedisperse(yy.T, dm, freqs, tsamp).T
            stokesI = xxdd + yydd
        else:
            stokesI = xxdd
        stokesI = np.ma.filled(stokesI, np.nan)
        lc = np.nansum(stokesI, axis=1)
        tie.append(lc)
    tie = np.vstack(tie)

    return tie


def signal2noise(
    data: np.ndarray,
    dms: np.ndarray,
    freqs: np.ndarray,
    dtsamp: float,
    reffreq: float | None = None,
    badchannels: set | list | np.ndarray | None = None,
    datarange: FInterval | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "median",
    background: tuple[float | dict, float | dict] = None,
    peak: bool = True,
    peak_interval: FInterval | None = None,
):
    """Calculate peak signal to noise values over a range of DM

    This calculates the light curve (dynamical spectrum summed across the channels) for
    Arguments
    ---------
    data : np.ndarray
        data containing freq on the y-axis and time on the x-axis

    dminterval : tuple[float, float]
        interval of the dispersion measure: start and stop

        An average DM is calculated from this range, which is then
        used in the calculation of the background: the data is
        dedispersed to this mean DM and the background is calculated,
        which is used for the bandpass correction.

    freqs : np.ndarray
        frequencies corresponding to the channel centers

    dtsamp : float
        sampling time interval in seconds

    reffreq: float or None

        reference frequency used for dispersion. If None, use the
        highest value of the given `freqs`.

    badchannels : set | list | np.ndarray | None, default=None
        numbers of channels to flag/ignore

    ndm : int, default=50
        Number of dm values to split the `dminterval` in to.

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

    bkg_method: string, "median" (default) or "mean"

        method to calculate a global background value from the
        background intervals.

    background: tuple of mean and standard deviation of the background
        values

        The tuple values can also be dicts. In that case, the keys are
        the polarization keys, (xx and yy), with the valuse the mean
        and standard deviation for those polarization parts. If the
        tuple elements are single values, but the input data contains
        multiple polarizations, it is assumed that the mean and
        standard deviation are the same for xx and yy.

    peak: bool, default True

        Optimize for the peak value. If False, optimize for the
        overall (integrated) light curve intensity.

    peak_interval: 2-tuple of float

        Fraction of the light curve where the peak is located. Setting this
        interval correctly reduces the amount of computation needed, and
        can speed up this function significantly.

        The background is still calculated for the full sample range.

    If the `background` argument is not `None`, `backgroundrange` and
    `bkg_method` are ignored. If `bkg_extra` is also set, the returned
    values identical to the given values.


    Returns
    -------
    Tuple of dm values and (peak) signal to noise ratios, both as a NumPy array.

    """

    if isinstance(backgroundrange[0], (float, int)):
        backgroundrange = [backgroundrange]

    dminterval = (dms[0], dms[-1])
    dm_center = (dminterval[0] + dminterval[1]) / 2

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

    if badchannels is not None:
        rowids = (
            list(badchannels)
            if not isinstance(badchannels, (list, np.ndarray))
            else badchannels
        )
        xx[:, rowids] = np.ma.masked
        if yy is not None:
            yy[:, rowids] = np.ma.masked

    logger.info("Dedispersing xx & yy")
    xx = dedisperse(xx.T, dm_center, freqs, dtsamp, reffreq=reffreq).T
    if yy is not None:
        yy = dedisperse(yy.T, dm_center, freqs, dtsamp, reffreq=reffreq).T

    # Bandpass correction
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
        logger.info("background calculation")
        xx_bkgmean, xx_bkgstd = calc_background(xx, backgroundrange, method=bkg_method)
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

    # xx and yy are now flagged, bandpass-corrected and dedispersed at the mean DM
    # This provides the starting point for the iteration through dmrange

    # If datarange is set, limit the data to this section.
    # This can provide faster iteration through the varying DM than (relative)
    # dedispersion of the full data set

    # Calculate overall standard deviation (single value)
    # for the light curve
    # by summing across the channels
    lightcurve = np.ma.filled(intensity, 0).sum(axis=1)
    idx_bkg = []
    nsamp = len(lightcurve)
    for bkgrange in backgroundrange:
        low = int(nsamp * bkgrange[0] + 0.5)
        high = int(nsamp * bkgrange[1] + 0.5)
        idx_bkg.append(np.arange(low, high))
    idx_bkg = np.concatenate(idx_bkg)
    lcstd = lightcurve[idx_bkg].std()

    if peak_interval:
        # Convert interval from fraction to integer samples
        n = xx.shape[0]
        # Note: +1.5 to have the +1 offset at the upper limit
        interval = slice(
            int(peak_interval[0] * n + 0.5), int(peak_interval[1] * n + 1.5)
        )
        xx = xx[interval, ...]
        if yy is not None:
            yy = yy[interval, ...]

    logger.info("Iterating over %d DMs from %.4f to %.4f", len(dms), dms[0], dms[-1])
    ratios = []
    for i, dm in enumerate(dms):
        logger.debug("dm = %.4f", dm)
        dm -= dm_center
        xxdd = dedisperse(xx.T, dm, freqs, dtsamp, reffreq=reffreq).T
        if yy is not None:
            yydd = dedisperse(yy.T, dm, freqs, dtsamp, reffreq=reffreq).T
            intensity = xxdd + yydd
        else:
            intensity = xxdd

        waterfall = np.ma.filled(intensity, 0)
        # Sum across frequencies to obtain the light curve
        lightcurve = waterfall.sum(axis=1)

        value = lightcurve.max() if peak else lightcurve.sum()
        ratio = value / lcstd
        ratios.append(ratio)

    return np.asarray(ratios)
