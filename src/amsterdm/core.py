import numpy as np

# According to Nimmo et al 2022, 10.1038/s41550-021-01569-9,
# this is also the value that is used in dspsr (digifil) and SFXC
DMCONST = 1.0 / 2.41e-4


__all__ = ["mask2d", "correct_bandpass", "dedisperse", "calc_intensity"]


def mask2d(data, rowids):
    """Mask rows in a two-dimensional array"""

    data = np.ma.masked_array(data, mask=False)
    data[:, rowids] = np.ma.masked
    return data


def correct_bandpass(
    data: np.ndarray | np.ma.MaskedArray,
    objrange: tuple[float, float] = (0.33, 0.66),
    method: str = "median",
    extra: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct for the individual channel bandpasses in a given data array

    Perform a bandpass correction: scale each channel with its
    background to correct for different sensitivities per channel.

    For each channel, a background level is estimated. This is done by
    selecting a set of time-sample columns outside of the columns that
    contain actual object of interest (the latter is given with
    `objrange` as a range in fraction of the total time-sample
    columns), then calculating an average, using one of three
    `method`s (mean, median or mode), and the background standard
    deviation (noise). The average is subtracted from the full channel
    values, then the channel is divided (normalized) by the standard
    deviation.

    Parameters
    ----------
    data : two-dimensional NumPy array
        data that needs be normalised. Usually contains frequency on
        the y-axis and time samples on the x-axis.

    objrange : two-tuple of floating point fractions between 0 and 1

        Fractional range along the time axis, where the actual object
        is located. Data outside these columns is used as the
        background for the estimation of mean/median/mode and standard
        deviation for the bandpass correction.

    method : one of "mean", "median" or "mode". Default "median".
        method to estimate the background level for each channel.

        Note that "mode" is not very applicable for continuously
        distributed data; and for normally distributed data, it will
        be the same value as the median or mean.

    extra : boolean
        if True, return the bandpass-corrected data, the background
        averages and the background standard devations. If False,
        return only the bandpass-corrected data.

    Returns
    -------
    Two-dimensional array with the bandpass corrected data

    """

    if method not in ["mean", "median", "mode"]:
        raise ValueError("method should be one of 'mean', 'median' or 'mode'")

    nsamp = data.shape[0]
    bkgrange = (int(nsamp * objrange[0]), int(nsamp * objrange[1]))

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
        bg_mean = np.ma.mean(bkg, axis=0)
    elif method == "median":
        bg_mean = np.ma.median(bkg, axis=0)
    elif method == "mode":
        bg_mean = np.empty(data.shape[1])
        for i in range(data.shape[1]):
            hist, bin_edges = np.histogram(bkg[:, i], bins=100)
            max_bin = np.argmax(hist)
            bg_mean[i] = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])
    else:  # we shouldn't be able to get here
        raise ValueError("method should be one of 'mean', 'median' or 'mode'")

    bg_std = np.ma.std(bkg, axis=0)
    data_sub = (data - bg_mean[None, :]) / bg_std[None, :]

    if extra:
        return data_sub, bg_mean, bg_std

    return data_sub


def dedisperse(
    data: np.ndarray | np.ma.MaskedArray,
    dm: float,
    freqs: np.ndarray,
    tsamp: np.ndarray,
    dmconst: float = DMCONST,
) -> np.ndarray:
    """Dedisperse a two-dimensional data set

    Parameters
    ----------
    data : 2d numpy array
        data containing freq on the y-axis and time on the x-axis
    dm : float
        Dispersion measure in units of pc / cc.
    freqs : 1d numpy array
        Array containing the channel frequencies in units of MHz
    tsamp : float
        sampling time in units of seconds.
    dmconst: float, default `DMCONST`
        dispersion constant in units of MHz^2 pc^-1 cm^3 s

    Returns
    -------
    ndata : 2d numpy
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
    badchan: set | list | np.ndarray | None = None,
    dm: dict[str, float | np.ndarray] | None = None,
    objrange: tuple[float, float] | None = None,
    bkg_extra: bool = False,
):
    """Returns the Stokes I / intensity parameter from the xx and yy data

    It will optionally correct for bad channels, bandpass and
    dispersion, if the relevant keyword argument is given.

    Parameters
    ----------

    data : dict with keys 'xx' and 'yy', and two-dimensional numpy
        array for value.

    badchan : set, list or array of channel indices to flag. The default of None
        means no flagging is done.

    objrange : two-tuple of floating point fractions between 0 and 1

        Fractional range along the time axis, where the actual object
        is located. Data outside these columns is used for the
        bandpass correction.

        The default of None indicates no bandpass correction is applied.

    dm : dict with keys "dm", "freq" and "tsamp", containing the
      disperson measure, the frequencies corresponding to the channels
      and the timestamps corresponding to the time-samples.

        Dedisperse the data for the given value. The default value of
        None means no dedispersion is applied.

    Returns
    -------
    Two-dimensional array with the Stokes intensity parameter.

    """

    xx = np.ma.array(data["xx"])
    yy = np.ma.array(data["yy"])

    if badchan:
        rowids = list(badchan) if isinstance(badchan, set) else badchan
        # xx = mask2d(xx, badchan)
        xx[:, rowids] = np.ma.masked
        yy[:, rowids] = np.ma.masked

    if dm:
        xx = dedisperse(xx.T, dm["dm"], dm["freq"], dm["tsamp"]).T
        yy = dedisperse(yy.T, dm["dm"], dm["freq"], dm["tsamp"]).T

    extra = {}
    if objrange:
        xx = correct_bandpass(xx, objrange, extra=bkg_extra)
        yy = correct_bandpass(yy, objrange, extra=bkg_extra)
        if isinstance(xx, tuple):
            xx, xx_bkgmean, xx_bkgstd = xx
            yy, yy_bkgmean, yy_bkgstd = yy
            extra["mean"] = xx_bkgmean + yy_bkgmean
            extra["std"] = np.sqrt(xx_bkgstd**2 + yy_bkgstd**2)

    intensity = xx + yy

    if extra:
        return intensity, extra

    return intensity
