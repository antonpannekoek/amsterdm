from types import EllipsisType

import numpy as np

# According to Nimmo et al 2022, 10.1038/s41550-021-01569-9,
# this is also the value that is used in dspsr (digifil) and SFXC
DMCONST = 1.0 / 2.41e-4


__all__ = [
    "mask2d",
    "measure_background",
    "correct_bandpass",
    "dedisperse",
    "calc_intensity",
    "bowtie",
]


def mask2d(data, rowids):
    """Mask rows in a two-dimensional array"""

    data = np.ma.masked_array(data, mask=False)
    data[:, rowids] = np.ma.masked
    return data


def measure_background(
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


def correct_bandpass(
    data: np.ndarray | np.ma.MaskedArray,
    datarange: tuple[float, float] = (0.3, 0.7),
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

    extra : bool, default=False
        if True, return the bandpass-corrected data, the background averages
        and the background standard devations. If False, return only the
        bandpass-corrected data.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    mean, std = measure_background(data, datarange, method)

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
    for i, bs in enumerate(bin_shift):
        newdata[i, :] = np.roll(data[i, :], bs)

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
        xx_bkgmean, xx_bkgstd = measure_background(xx, datarange)
        yy_bkgmean, yy_bkgstd = measure_background(yy, datarange)

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


def bowtie(
    data: np.ndarray,
    dm: tuple[float, float],
    freqs: np.ndarray,
    tsamp: float,
    badchannels: set | list | np.ndarray | None = None,
    datarange: tuple[float, float] = (0.3, 0.7),
    ndm: int = 50,
    samplerange: slice | EllipsisType = Ellipsis,
) -> np.ndarray:
    """
    Create the data for a bowtie plot: varying DM versus time/samples

    Parameters
    ----------
    data : np.ndarray
        data containing freq on the y-axis and time on the x-axis

    dm : tuple[float, float]
        range of the dispersion measure: start and stop

    freqs : np.ndarray
        frequencies corresponding to the channel centers

    tsamp : float
        sampling time interval in seconds

    badchannels : set | list | np.ndarray | None, default=None
        numbers of channels to flag/ignore

    datarange : tuple[float, float], default=(0.3, 0.7)
        Sample fractions within which the burst is located (after
        dedispersion): lower and upper fraction. The remainder area is used for
        background determination.

        This section applies to the full data range; ``samplerange`` is ignored for
        the background calculation.

        The background is calculcated once for the central DM (thus for
        ``(dm[0] + dm[1]) / 2``), so the datarange should be for the particular
        DM; all other DM samples use the same background value.

    ndm : int, default=50
        Number of DM samples along the y-axis

    samplerange : slice | EllipsisType, default=Ellipsis
        Section of data on the "samples" axis to be used.

        For the background calculationt, the full data section is used.

    Returns
    -------
    np.ndarray
    """

    dm_center = (dm[0] + dm[1]) / 2

    # Copy the data, flag bad channels and get a background estimate for the central DM
    # After that, perform a bandpass correction and dedisperse to the central DM
    # This provides the starting point for varying DM
    # This assumes linear XX and YY polarization channels

    xx = np.ma.array(data[:, 0, :])
    yy = np.ma.array(data[:, 1, :])

    if badchannels:
        rowids = (
            list(badchannels)
            if not isinstance(badchannels, (list, np.ndarray))
            else badchannels
        )

        xx[:, rowids] = np.ma.masked
        yy[:, rowids] = np.ma.masked

    xx = dedisperse(xx.T, dm_center, freqs, tsamp).T
    yy = dedisperse(yy.T, dm_center, freqs, tsamp).T

    # Bandpass correction
    mean, std = measure_background(xx, datarange)
    xx = (xx - mean[None, :]) / std[None, :]
    mean, std = measure_background(yy, datarange)
    yy = (yy - mean[None, :]) / std[None, :]

    # Limit the data range
    xx = xx[samplerange, :]
    yy = yy[samplerange, :]
    # xx and yy are now flagged, bandpass-corrected and dedispersed at the mean DM
    # This provides the starting point for the iteration through dmrange

    # dmrange is relatively the mean DM
    dmrange = np.linspace(dm[0], dm[1], ndm) - dm_center

    tie = []
    for dm in dmrange:
        xxdd = dedisperse(xx.T, dm, freqs, tsamp).T
        yydd = dedisperse(yy.T, dm, freqs, tsamp).T
        stokesI = xxdd + yydd
        stokesI = np.ma.filled(stokesI, np.nan)
        lc = np.nansum(stokesI, axis=1)
        tie.append(lc)
    tie = np.vstack(tie)

    return tie
