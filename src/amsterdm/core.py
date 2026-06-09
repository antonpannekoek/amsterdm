"""This module contains core utilities for handling FRB data

The input data
--------------

The data is generally assumed to be a two- or three-dimensional NumPy
array of "intensity" data (simply the signal). If three-dimensional,
the central dimension is assumed to be the polarization channel, and
should have a size of 1 (only Stokes I), 2 (e.g. xx and yy) or
4. Since there is no header available for the NumPy array to indicate
what type of polarization channel(s) are included, the user will have
to pay attention to this themselves; for example, if the data is
three-dimensional Stokes I, Q, U and V data, it is probably best to
change to use only the I data, which is applicable for most cases.

The first dimension (axis 0) should be the time data, and the last
dimension (axis 1 or 2, for two- or three-dimensional data,
respectively) should be the channel / frequency data.

Other inputs often required are the frequencies (`freqs)`, which is a
list or array of frequencies, matching one-to-one with the
channels. The channels correspond to the indices into the frequency
dimension of the data, that is, the last dimension. Frequency
generally matches the channels linearly, though the relationship may
be in reverse (that is, increasing frequencies follow decreasing
channels).

The unit of frequencies is assumed to be megahertz, MHz.

The time axis, axis 0, is assumed to be linear, and the first index
(index 0) is simply assumed to be 0. The sampling time, `tsamp`, is
the time interval and is often required as an input. The unit of the
sampling time is assumed to be milliseconds, ms.

Dispersion measure, `dm`, is often input, in units of MHz^2 cm^3 pc^-1
ms; the dispersion constant, K, is set to a fixed value, of 2.41 *
10^-7 (Nimmo et al 2022, DOI:10.1038/s41550-021-01569-9), see also the
`constants` module. Since this constant, or rather 1/K, is multiplied
by `dm`, the latter can be unitless.


Units
-----

As noted above, the units assumed in the code are as follows:

- time: millisecond
- frequency: megahertz

The current version of AmsterDm does not use units internally. Future
versions may use e.g. AstroPy's `units` module for consistency and
safer calculations.

"""

import logging

from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian1D
from astropy import units
from astropy.units import Quantity
import numpy as np

from .constants import DEFAULT_BACKGROUND_RANGE, DMCONST, DMUNIT, Array
from .utils import FInterval, QInterval


__all__ = [
    "bowtie",
    "calc_background",
    "calc_intensity",
    "calc_lightcurve",
    "calc_lightcurve_from_waterfall",
    "create_dynspectrum",
    "dedisperse",
    "downsample",
    "findpeaklc",
    "findrangelc",
    "fit_ratios",
    "flag",
    "signal2noise",
    "upsample",
]


logger = logging.getLogger(__package__)


def ensure_quantities(
    tsamp: None | float | Quantity = None,
    freqs: None | float | Quantity = None,
    reffreq: None | float | Quantity = None,
    dm: None | float | Quantity = None,
    dminterval: None | FInterval | QInterval = None,
) -> tuple[Quantity, Quantity, Quantity, Quantity, QInterval]:
    """Helper function to ensure inputs are always in a correct quantity

    Also sets the reference frequency `reffreq` to the maximum of the frequencies,
    if `freqs` is not None and `reffreq` is None.

    Returns a tuple of all the input values, in order of the input arguments.

    If any input value is `None`, it is not converted (excepted for `reffreq`),
    and `None` is returned for its value in the returned tuple.

    For tsamp, the default unit if not provided is millisecond.

    For freqs and reffreq, the default unit if not provided is MHz.

    """

    if tsamp is not None and not isinstance(tsamp, Quantity):
        tsamp *= units.millisecond
    if freqs is not None:
        # Take care not to change the original input list/array
        freqs_ = freqs[:]
        if not isinstance(freqs, Quantity):
            freqs_ *= units.MHz
    if reffreq is None and freqs is not None:
        # Use maximum frequency as reference frequency
        reffreq = np.max(freqs)
    if reffreq is not None and not isinstance(reffreq, Quantity):
        reffreq *= units.MHz
    if dm is not None and not isinstance(dm, Quantity):
        dm *= DMUNIT
    if dminterval is not None and not isinstance(dminterval[0], Quantity):
        dminterval = (dminterval[0] * DMUNIT, dminterval[1] * DMUNIT)

    return (tsamp, freqs_, reffreq, dm, dminterval)


def _format_data(
    data: Array | dict[str, Array], poltype: str = ""
) -> tuple[Array, str]:
    """Format input data to into a consistent 3d array

    This function takes the input data and reformats it into a
    consistent three-dimensional array. This output array may have a
    dimension of 1 for the second axis, the polarization channel axis,
    in case the input was simply two-dimensional (or three-dimensional
    with a dimension 1 second axis).

    `poltype` sets the polarization type and can be "iquv", "xy" or
    "ab", but in most cases is deduced from the input (if left at its
    default of an empty string). If the deduced polarization type does
    not match the `poltype` argument, a `ValueError` is raised.

    The input can be a two-dimensional array, a three-dimensional
    array or a dict of various polarization channels: each dict value
    is a two-dimensional array for that specific channel. The keys
    indicate the polarization type of the input, and can be matching
    keys "i", "q", "u", "v", "xx", "yy", "aa", "bb"; all format
    strings and keys are case independent. There is a minimal check
    that if "xx" is present, "yy" should be present (and vice versa),
    and the same for "aa" and "bb"; for "q", "u" and "v", there should
    at least be a "i" key present as well. From the keys, the
    polarization type is deduced, and if `poltype` is not empty, a
    check is made to see if this matches.

    If the input is three-dimensional array, the deduced polarization
    type depends on the size of the polarization channel dimension
    (the second, center, axis): if it's 1, Stokes I is assumed; if 4,
    Stokes I, Q, U and V is assumed; if 2, `poltype` should be one of
    "xy" or "ab" to give the polarization type.

    Returns
    -------
        A tuple of a three-dimensional array and a string:
        - The three-dimensional array matches the input data, and its
          dimensions are (samples/time, polarization,
          channel/frequency)
        - the string describes the (deduced) polarization type

    Raises
    ------
    ValueError
        - in case the deduced format (from the polarization channel
          dimension or the dict keys) does not match the given
          `poltype` (if not empty).
        - in case of a missing key (e.g., "yy" missing when "xx" given)
        - in case of incompatible keys (e.g., "xx" and "bb")
        - for a dict input: when any value array is not two-dimensional
        - for a dict input: when the value arrays are not the same shape

    """

    # List of valid `poltype`s and their matching dict keys
    polkeys = {
        "xy": ("xx", "yy"),
        "ab": ("aa", "bb"),
        "i": ("i",),
        "iquv": ("i", "q", "u", "v"),
        "iq": ("i", "q"),
        "iu": ("i", "u"),
        "iv": ("i", "v"),
        "iqu": ("i", "q", "u"),
    }

    poltype = poltype.lower()
    if poltype and poltype not in polkeys:
        raise ValueError("invalid polarization type")

    if isinstance(data, dict):
        keys = [key.lower() for key in data.keys()]
        pairs = [
            ("xx", "yy"),
            ("yy", "xx"),
            ("aa", "bb"),
            ("bb", "aa"),
            ("q", "i"),
            ("u", "i"),
            ("v", "i"),
        ]
        for key1, key2 in pairs:
            if key1 in keys:
                if key2 not in keys:
                    raise ValueError(f"missing '{key2}' key to match '{key1}'")
        if poltype:  # validate polarization type
            if set(polkeys[poltype]) != keys:
                raise ValueError("`poltype` does not match the input dict keys")
        shape = None
        for arr in data.values():
            if not isinstance(arr, (np.ndarray, np.ma.MaskedArray)):
                raise ValueError("dict value is not an array")
            if arr.ndim != 2:
                raise ValueError("dict value is not two-dimensional")
            if not shape:
                shape = arr.shape
            elif arr.shape != shape:
                raise ValueError("dict values are inconsistent in shape")

        # reverse the dict, to reformat the data and set an output `poltype`
        inv = {value: key for key, value in polkeys.items()}
        # Note: polarization names are conveniently in alphabetical order
        skeys = tuple(sorted(keys))
        poltype = inv[skeys]
        ndim = len(poltype)
        shape = data[skeys[0]].shape
        shape = (shape[0], ndim, shape[1])
        # Can we avoid all the copying here?
        fmtdata = np.empty(shape)
        for i, key in enumerate(skeys):
            fmtdata[:, i, :] = data[key][...]

    else:
        if not isinstance(data, (np.ndarray, np.ma.MaskedArray)):
            raise ValueError("data is not an array")
        if data.ndim not in (2, 3):
            raise ValueError("data is not two- or three-dimensional")
        shape = data.shape

        if data.ndim == 3:
            poldim = shape[1]
            if poltype:
                if len(poltype) != poldim:
                    raise ValueError(
                        "polarization type does not match the number "
                        "of polarization channels"
                    )
            else:
                # Deduce a default polarization type from the dimension
                if poldim == 1:
                    poltype = "i"
                elif poldim == 2:  # to do: raise error as this is ambiguous?
                    poltype = "xy"
                elif poldim == 3:
                    poltype = "iqu"
                elif poldim == 4:
                    poltype = "iquv"
                else:
                    raise ValueError("incorrect number of polarization channels")
            fmtdata = data

        else:  # two-dimensional data
            if poltype and poltype != "i":
                raise ValueError("incorrect polarization type for two-dimensional data")
            # Turn the 2D data into 3D data for consistency
            fmtdata = data.reshape(shape[0], 1, shape[1])
            poltype = "i"

    return fmtdata, poltype


def downsample(
    data: Array,
    factor: int,
    remainder: str = "droptail",
    method: str = "mean",
) -> Array:
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

    Parameters
    ----------
    data : Array
        Multi-dimensional input data array.
    factor : int
        Factor to downsample by.
    remainder : str, default "droptail"
        What to do with the remainder if `factor` does not divide the
        first axis size fully, i.e., there is a remainder. The
        remainder can be dropped, either from the start ("drophead" of
        the data or from the end ("droptail"), measured along the
        first axis; or the remainder can be added to the the first
        "bin" ("addhead") or the last "bin" ("droptail") along the
        first axis.
    method: str, default "mean"
        Whether to average ("mean") or sum ("sum") each bin to its
        downsampled value.

    Returns
    -------
    Downsampled array : Array

    Raises
    ------
    ValueError
        - for incorrect data dimensions (less than 2)
        - for an incorrect `factor` (less than 1 or non-integer)
        - for an incorrect `remainder` value
        - for an incorrect `method`

    Examples
    --------
    >>> a =  np.arange(10)
    >>> downsample(a, 1)
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> downsample(a, 2)
    array([0.5, 2.5, 4.5, 6.5, 8.5])
    >>> a = a.reshape(5, 2)
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> downsample(a, 2)
    array([[1., 2.],
           [5., 6.]])
    >>> downsample(a, 5)
    array([[4., 5.]])
    >>> downsample(a, 3)
    array([[2., 3.]])
    >>> downsample(a, 3, remainder="drophead")
    array([[6., 7.]])
    >>> downsample(a, 3, remainder="addtail")
    array([[4., 5.]])
    >>> downsample(a, 5, method="sum")
    array([[20, 25]])
    """

    if factor < 1 or not isinstance(factor, int):
        raise ValueError("'factor' should be a positive integer")
    if method not in ("mean", "sum"):
        raise ValueError("'method' should be one of 'mean' or 'sum'")
    if remainder not in ("droptail", "addtail", "drophead", "addhead"):
        raise ValueError(
            "'remainder' should be one of 'droptail', 'addtail', 'drophead' or 'addhead'"
        )

    n = data.shape[0]
    if n <= factor:
        # Combine all bins
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
    data: Array,
    factor: int,
) -> Array:
    """Rebin the data to a higher resolution along the first
    (sample/time) axis

    Sample bins are simply split into `factor` new bins, with the same
    value as that of the original bin.

    Under the hood, this simply uses `numpy.repeat` for the first
    axis.

    Parameters
    ----------
    data : array
        Input array
    factor : int
        Factor to upsample by

    Returns
    -------
        Upsampled array : array

    Raises
    ------
    ValueError
        For an incorrect `factor` (less than 1 or non-integer)

    Examples
    --------
    >>> a = np.arange(4)
    >>> upsample(a, 1)
    array([0, 1, 2, 3])
    >>> upsample(a, 2)
    array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> upsample(a, 2)
    array([[0, 1],
           [0, 1],
           [2, 3],
           [2, 3]])
    """

    if factor < 1 or not isinstance(factor, int):
        raise ValueError("'factor' should be a positive integer")

    return np.repeat(data, factor, axis=0)


def findpeaklc(
    data: Array,
    searchrange: tuple[float, float] = (0, 1),
) -> int:
    """Find the peak of the light curve, within an optional search range

    data: the one-dimensional light curve intensity data

    searchrange: a 2-tuple of floats
        Fractional start and end of the search range

    This simply returns the index of the maximum of `data`, potentially
    restricted to a section of the data by `searchrange`.

    Parameters
    ----------
    data : array
        The one-dimensional light curve intensity data
    searchrange : tuple
        A 2-tuple of floats indicating the fractional start and end of
        the search range

    Returns
    -------
    int
        the index into data of the peak

    Raises
    ------
    ValueError
        For an invalid search range
    """

    if searchrange[0] < 0 or searchrange[1] > 1 or searchrange[0] >= searchrange[1]:
        raise ValueError("invalid searchrange; should be an interval between 0 and 1")

    n = len(data)
    low, high = int(searchrange[0] * n + 0.5), int(searchrange[1] * n + 0.5)
    index = np.argmax(data[low:high]) + low
    return index


def findrangelc(
    data: Array,
    kappa: float = 10,
    minkappa: float = 3,
    window: int = 7,
    maxiter: int = 10,
    minvalues: int = 10,
    searchrange: tuple[float, float] = (0, 1),
    bkg: tuple[float, float] | None = None,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Find the range of the active light curve.

    Description
    -----------

    The algorithm first smooths the light curve by using a rolling
    average with `window` size. No smoothing is done if `window` is
    negative.

    It then iteratively (up to `maxiter` times):
      - calculates a mean value
      - find all values below that mean
      - removes all non-found values (i.e., outlier peaks)

    Iteration stops when there are less than `minvalues` (default of
    10) values left or `maxiter` iterations have been reached.

    The remaining data are seen as the background. It takes the
    indices of these remaining data, and calculates a median and
    standard deviation from the non-smoothed data for this selection
    of indices; this is used as a first estimate for the background
    value and its noise.

    If the `bkg` argument is given (as two values, the median and
    standard deviation), the above calculation is skipped.

    It then finds all values in the smoothed data that are `kappa`
    times noise above the background. The relevant indices are
    combined into sections, and each of these sections are extended on
    both sides to a `minkappa` times the noise above the
    background. The latter step is done separately, so that incidental
    low-sigma spikes above the background are not included, only when
    adjacent to a larger foreground region.

    If `searchrange` is given, the above calculation is done in the
    given range.

    These sections then define the foreground area where there is an
    active light curve.

    The sections are then returned, as a list of 2-tuples with start
    and end indices.

    Parameters
    ----------
    data : Array
        The one-dimensional light curve intensity data
    kappa : float, default 10
        Find peaks that are `kappa` times the noise above
    minkappa : float, default 3
        minimal noise value to be included next to the peak areas
    maxiter : int, default 10
        Maximum number of iterations to find an average (by
        iteratively removing outlier peaks).
    minvalues : int, deafult 10
        Minimum number of data points to keep when iteratively
        determining for an average
    window : integer, default 7
        number of bins to use in the rolling average. Use -1 for no
        rolling average
    searchrange : a 2-tuple of floats, default (0, 1)
        Fractional start and end of the search range. Note that the
        background calculation is odne for the full range.
    bkg : 2-tuple, default None
        If known, the background value and its standard deviation can
        be given here. If not given (bkg=None), the background is
        calculated as described above.

    Returns
    -------
       A tuple of 2 items:
       - A list of 2-tuples of integers. These represent the start and end indices
         of sections where the light curve is active.
       - A tuple of two floating point values: the estimated
         background value and standard deviation

    Raises
    ------
    ValueError
        If the data is not one-dimensional

    """

    if data.ndim != 1:
        raise ValueError("input `data` should be one-dimensional")

    # Smooth the data with a window
    if window > 1:
        sdata = np.convolve(data, np.ones(window), mode="same") / window
    else:
        sdata = data

    if not bkg:
        selection = np.ones(len(sdata), dtype=bool)
        for i in range(maxiter):
            mean = sdata[selection].mean()
            selection = selection & (sdata < mean)
            if selection.sum() < minvalues:
                break
        bkgval = np.ma.median(data[selection])
        bkgstd = data[selection].std()
    else:
        bkgval, bkgstd = bkg

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

    return sections, (bkgval, bkgstd)


def calc_background(
    data: Array,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    method: str | None = "mean",
) -> tuple[Array, Array]:
    """Return background and its standard deviation for each channel

    Assumes any dispersion correction has already been done, so that
    the signal is not smeared out across background sections.

    For each individual channel (if any, that is, for
    three-dimensional data), a background level is estimated. This is
    done by averaging over the first axis in the `backgroundrange`
    intervals. The method for the "average" is given by `method`, and
    can be "mean", "median" or "mode". A standard deviation is
    calculated from the background (this ignore `method`).

    Parameters
    ----------
    data : Array
        data that needs be normalised. Usually contains frequency on the y-axis
        and time samples on the x-axis.

    backgroundrange : iterable of 2-tuples, or None
        Iterable (e.g., list) of ranges as fractions of the sample
        dimension of the data, that is, each iterable item contains a
        begin and end fraction of the first dimension of the data that
        corresponds to a background area

        If None, uses the full range for the background calculation.

    method : str, default="mean"
        method to estimate the background level for each channel.

        Note that "mode" is not very applicable for continuously distributed
        data; and for normally distributed data, it will be the same value as
        the median or mean.

        if `method` is "none" or `None`, no background is calculated, and
        a tuple of 2 zero arrays is returned

    Returns
    -------
    Tuple of 2 Arrays
        The background value  and standard deviation across all frequency channels

    Raises
    ------
    ValueError
        - in case of an invalid `method`
        - in case of an invalid background section (outside of (0, 1))

    """

    if method not in ["mean", "median", "mode", "none", None]:
        raise ValueError("method should be one of 'mean', 'median', 'mode' or 'none'")

    if method in ["none", None]:
        n = data.shape[-1]
        return np.zeros(n), np.zeros(n)

    if backgroundrange is None:
        backgroundrange = [[0, 1]]
    else:
        if isinstance(backgroundrange[0], (float, int)):
            backgroundrange = [backgroundrange]

    nsamp = data.shape[0]
    idx_bkg = []
    for bkgrange in backgroundrange:
        if bkgrange[0] < 0 or bkgrange[1] > 1 or bkgrange[1] <= bkgrange[0]:
            raise ValueError("incorrect background range")
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
        mean = np.ma.empty(data.shape[1])
        for i in range(data.shape[1]):
            hist, bin_edges = np.histogram(bkg[:, i], bins=100)
            max_bin = np.argmax(hist)
            mean[i] = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])
    else:  # we shouldn't be able to get here
        raise ValueError("method should be one of 'mean', 'median' or 'mode'")
    std = np.ma.std(bkg, axis=0)

    if not isinstance(data, np.ma.MaskedArray):
        # Turn mean and std back to plain arrays
        mean = mean.filled(np.nan)
        std = std.filled(np.nan)

    return mean, std


def correct_bandpass(
    data: Array,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    method: str = "mean",
) -> tuple[Array, Array, Array]:
    """Correct for the individual channel bandpasses in a given data array

    Perform a bandpass correction: scale each channel with its background to
    correct for different sensitivities per channel.

    Description
    -----------

    For each channel (if available, in case of three-dimensional
    data), a background level is estimated; see `calc_background` for
    a description.  The background is subtracted from the full channel
    values, then the channel is divided (normalized) by the background
    standard deviation.

    Parameters
    ----------
    data : Array
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

    method : str, default="mean"
        method to estimate the background level for each channel.

        Note that "mode" is not very applicable for continuously distributed
        data; and for normally distributed data, it will be the same value as
        the median or mean.

    Returns
    -------
    tuple[Array, Array, Array]
        Tuple of the bandpass corrected array, the mean background,
        and the background standard deviation

    """

    mean, std = calc_background(data, backgroundrange, method)

    # Bandpass correction
    data_sub = (data - mean[None, ...]) / std[None, ...]

    return data_sub, mean, std


def dedisperse(
    data: Array,
    freqs: list | np.ndarray | Quantity,
    tsamp: float | Quantity,
    dm: float | Quantity,
    reffreq: float | Quantity | None = None,
    dmconst: Quantity = DMCONST,
) -> Array:
    """Dedisperse a two-dimensional data set

    Dedispersion is done using `numpy.roll` for the individual "rows",
    i.e., per frequency, which shifts (and wraps around) the data per
    row. The necessary amount to shift for each row is calculated from
    the reference frqeuency, the frequencies, the dispersion measure
    `dm` and the dispersion constant; the resulting values are rounded
    to integers.

    Parameters
    ----------
    data : Array
        data containing freq on the y-axis (outer axis) and time on
        the x-axis (inner axis)
    freqs : np.ndarray, Quantity
        Array containing the channel frequencies in units of MHz
    tsamp : float, Quantity
        sampling time in units of milliseconds.
    dm : float
        Dispersion measure in units of pc / cc.
    reffreq : float, optional
        The reference frequency in MHz, to tie the dedispersion to. If not
        given, the maximum of the input frequency range (`freqs`) is used.
    dmconst : float, default=DMCONST
        dispersion constant in units of MHz^2 pc^-1 cm^3 s

    Returns
    -------
    ndata : Array
        dedispersed data

    Raises
    ------
    ValueError
        if the length of `freqs` doesn't match the first axis of `data`.

    """

    if data.shape[-1] != len(freqs):
        raise ValueError(
            "`freqs` length does not match the last axis of the data array"
        )

    (tsamp, freqs, reffreq, dm, _) = ensure_quantities(tsamp, freqs, reffreq, dm, None)

    if dm.value == 0:
        return data.copy()

    # calculate time shifts and convert to bin shifts
    time_shift = dmconst * dm * (reffreq**-2.0 - freqs**-2.0)
    bin_shifts = (time_shift / tsamp).decompose().value
    # round to nearest integer
    bin_shifts = np.rint((time_shift / tsamp).decompose()).value.astype(np.int64)
    # Assert that there is a shift for each channel / frequency
    assert len(bin_shifts) == data.shape[-1]

    # We are using transpose here, since then we can assign
    # to the rows, which is faster.
    # Conveniently, the "transpose" of three-dimensional data
    # switches the first and last dimension, which is want we want
    # init empty array to store dedisp data in
    # Note that `newdata` is transposed, so that
    # the shifted data can be assigned per row
    # (which should be faster)
    tdata = data.T
    newdata = np.empty_like(tdata)
    if isinstance(data, np.ma.MaskedArray):
        # Assert that `empty_like` copies the mask from `data`
        np.testing.assert_equal(newdata.mask, tdata.mask)

    # dedisperse by rolling back the channels
    if tdata.ndim == 3:
        for j in range(tdata.shape[1]):  # polarization channels
            for i, shift in enumerate(bin_shifts):  # frequency channels
                newdata[i, j] = np.roll(tdata[i, j], shift)
    else:
        for i, shift in enumerate(bin_shifts):
            newdata[i] = np.roll(tdata[i], shift)

    return newdata.T


def flag(data: Array, badchannels: set | list | np.ndarray) -> np.ma.MaskedArray:
    """Flag bad channels by masking the corresponding channels rows

    Parameters
    ----------

    data: Array
        Input data, two- or three-dimensional. Data is flagged along
        the last dimension.

    badchannels: list, set or array of integers, optional
        List (or set or ndarray) of bad channels to flag. These are
        integers indices into the channel/frequency dimension of
        `data` (the last dimension/axis). Note that the channel order is
        applied: if the frequencies are in reverse order compared to
        the frequency (i.e., delta-frequency is negative),
        `badchannels` follows the channel order. If not given or
        `None`, no channels are flagged.

        Flagging is done by masking the relevant columns in the `data`
        array, turning the data array into a masked array. The bad
        channels are assumed to be the same for all polarizations.

        Existing flagged channels (an existing mask) will be kept, and
        combined with the new flags; that is, the flags are or-ed
        together.

    Returns
    -------
        np.ma.MaskedArray, even if no channels were flagged

    """

    mdata = np.ma.array(data)

    # Allow a set of bad channels as input, but we can't index
    # an array with a set, so convert to a list
    rowids = (
        list(badchannels)
        if not isinstance(badchannels, (list, np.ndarray, np.ma.MaskedArray))
        else badchannels
    )

    # Existing masks are kept, and the new mask is just added on top
    # of that.
    mdata[..., rowids] = np.ma.masked

    return mdata


def calc_intensity(
    data: dict[str, Array] | Array,
    freqs: np.ndarray,
    tsamp: float,
    dm: float,
    reffreq: float | None = None,
    backgroundrange: FInterval | tuple[FInterval] | None = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
) -> tuple[Array, tuple[float, float]]:
    """Returns the Stokes I / intensity parameter from the xx and yy data

    .. deprecated::
        use `create_dynspectrum` instead.

    It will optionally correct for bad channels, bandpass and dispersion, if
    the relevant keyword argument is given.

    Currently, only the xx/yy is supported.

    Parameters
    ----------
    data : Array or a dict of str, Array
        This is a single three-dimensional array, or a two-dimensional
        array, or a dictionary with keys of "xx" and "yy" and
        two-dimensional arrays (time and frequency) as values. For the
        three-dimensional array, the second dimension should be the
        "polarization" channel, in this case only the xx/yy data
        (i.e., of size 2).

    freqs: ndarray
        Frequencies corresponding to the data

    tsamp: float
        Sampling time in milliseconds

    dm: float
        Dispersion measure. Zero or `None` will skip the dedispersion step

    reffreq: float, option.
        Reference frequency for the dedispersion. If not given (or
        `None`), the maximum frequency from the `freqs` array is used.

    backgroundrange : iterable of 2-tuple of float, optional
        Fractional range along the time axis where the background
        should be measured. See `calc_background` for details.

        Set to `None` to not apply a bandpass correction (no background
        is calculated).

        The bandpass correction is calculated separately for each
        polarization (if applicable), but with identical
        `backgroundrange` s for each polarization channel.

    bkg_method: str, optional
        What method to use for the background calculation. One of
        "mean", "median" or "mode". See `calc_background` for details.

        if `method` is "none" or `None`, no background is calculated, and
        a tuple of 2 zero arrays is returned

    Returns
    -------
    tuple of Array, (float, float)
        A 2-tuple of the intensity data and the background. The
        background consists of a 2-tuple of the background mean (its
        value) and the background standard deviation.
        If backgroundrange is `None`, the background returned will be (0, 0)

    """

    if isinstance(data, dict):
        xx = data["xx"]
        yy = data["yy"]
    elif isinstance(data, (np.ndarray, np.ma.MaskedArray)):
        if data.ndim == 3:
            xx = data[:, 0, :]
            yy = data[:, 1, :]
        elif data.ndim == 2:
            xx = data
            yy = None
        else:
            raise ValueError("data is not 2- or 3-dimensional")
    else:
        raise ValueError("data is not a single array or a dict of arrays")

    if yy is not None:
        if xx.shape != yy.shape:
            raise ValueError("'xx' and 'yy' channels do no match in dimensions")
    if xx.shape[-1] != len(freqs):
        raise ValueError("`freqs` length does not match the last axis of `data`")

    if dm:
        xx = dedisperse(xx, freqs, tsamp, dm)
        if yy is not None:
            yy = dedisperse(yy, freqs, tsamp, dm)

    bkg_mean = bkg_std = np.zeros(len(freqs))  # default values
    if bkg_method not in ["none", None]:
        xx_bkgmean, xx_bkgstd = calc_background(xx, backgroundrange, bkg_method)
        if yy is not None:
            yy_bkgmean, yy_bkgstd = calc_background(yy, backgroundrange, bkg_method)

        # Bandpass correction
        xx = (xx - xx_bkgmean[None, :]) / xx_bkgstd[None, :]
        if yy is not None:
            yy = (yy - yy_bkgmean[None, :]) / yy_bkgstd[None, :]

        bkg_mean = xx_bkgmean + yy_bkgmean if yy is not None else xx_bkgmean
        bkg_std = np.sqrt(xx_bkgstd**2 + yy_bkgstd**2) if yy is not None else xx_bkgstd

    intensity = xx + yy if yy is not None else xx

    return intensity, (bkg_mean, bkg_std)


# To do: generalize to any (valid) number of
# polarization channels
def _create_dynspectra(
    data: Array | dict[str, Array],
    freqs: list | np.ndarray | Quantity,
    tsamp: float | Quantity,
    dm: float | Quantity,
    reffreq: float | Quantity | None = None,
    backgroundrange: FInterval | tuple[FInterval] | None = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str | None = "mean",
    background: tuple[float | dict, float | dict] | None = None,
) -> dict[str, tuple[Array, float, float]]:
    """Create a dynamical spectrum for each polarization channel

    See `create_dynspectrum` for the details on the arguments

    Returns one or two dynamical spectra

    """

    data, poltype = _format_data(data)
    data = np.squeeze(data)

    yy = None
    if data.ndim == 2:
        xx = np.ma.array(data)
    elif data.shape[1] != 2:
        if poltype == "iquv":
            xx = np.ma.array(data[:, 0, :])
        else:
            raise ValueError("second (polarization) dimension has incorrect size")
    else:
        xx = np.ma.array(data[:, 0, :])
        yy = np.ma.array(data[:, 1, :])

    (tsamp, freqs, reffreq, dm, _) = ensure_quantities(tsamp, freqs, reffreq, dm, None)

    if len(freqs) != xx.shape[-1]:
        raise ValueError(
            "`freqs` length does not match the last axis of the data array"
        )

    if dm.value:
        xx = dedisperse(xx, freqs, tsamp, dm, reffreq=reffreq)
        if yy is not None:
            yy = dedisperse(yy, freqs, tsamp, dm, reffreq=reffreq)

    if background:
        # Use a given background
        bkgmean, bkgstd = background
        if isinstance(bkgmean, dict):
            # Separate background values for xx and yy
            xx_bkgmean = bkgmean["xx"]
            yy_bkgmean = bkgmean["yy"]
        elif bkgmean.ndim == 2:
            xx_bkgmean = bkgmean[0]
            yy_bkgmean = bkgmean[1]
        else:
            xx_bkgmean = yy_bkgmean = bkgmean
        if isinstance(bkgstd, dict):
            # Separate background values for xx and yy
            xx_bkgstd = bkgstd["xx"]
            yy_bkgstd = bkgstd["yy"]
        elif bkgstd.ndim == 2:
            xx_bkgstd = bkgstd[0]
            yy_bkgstd = bkgstd[1]
        else:
            xx_bkgstd = yy_bkgstd = bkgstd

        # Expand any scalar to a 1D array
        if isinstance(xx_bkgmean, (int, float)) or xx_bkgmean.ndim == 0:
            xx_bkgmean = xx_bkgmean * np.ones(len(freqs))
        if isinstance(xx_bkgstd, (int, float)) or xx_bkgstd.ndim == 0:
            xx_bkgstd = xx_bkgstd * np.ones(len(freqs))
        if isinstance(yy_bkgmean, (int, float)) or yy_bkgmean.ndim == 0:
            yy_bkgmean = yy_bkgmean * np.ones(len(freqs))
        if isinstance(yy_bkgstd, (int, float)) or yy_bkgstd.ndim == 0:
            yy_bkgstd = yy_bkgstd * np.ones(len(freqs))
    else:
        # Calculate the background from the data
        xx_bkgmean, xx_bkgstd = calc_background(xx, backgroundrange, method=bkg_method)
        if yy is not None:
            yy_bkgmean, yy_bkgstd = calc_background(
                yy, backgroundrange, method=bkg_method
            )
        else:
            yy_bkgmean = np.zeros(len(freqs))
            yy_bkgstd = np.zeros(len(freqs))

    # Perform the bandpass correction using the background
    # Note: since xx is a masked array, any division by zero
    # (for any stddev point that is zero)) will result in those entries being masked
    # This is convenient `np.ma` behaviour.See
    # https://numpy.org/doc/stable/reference/maskedarray.generic.html#operations-on-masked-arrays
    if bkg_method not in ["none", None]:
        xx = (xx - xx_bkgmean[None, :]) / xx_bkgstd[None, :]
        if yy is not None:
            yy = (yy - yy_bkgmean[None, :]) / yy_bkgstd[None, :]

    return {"xx": [xx, xx_bkgmean, xx_bkgstd], "yy": [yy, yy_bkgmean, yy_bkgstd]}


def create_dynspectrum(
    data: Array | dict[str, Array],
    freqs: list | np.ndarray | Quantity,
    tsamp: float | Quantity,
    dm: float | Quantity,
    reffreq: float | Quantity | None = None,
    backgroundrange: FInterval | tuple[FInterval] | None = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str | None = "mean",
    background: tuple[float | dict, float | dict] | None = None,
    combine: str = "mean",
) -> tuple[Array, tuple[float, float]]:
    """Returns a dynamical spectrum with the Stokes I / intensity
    parameter from the input data data array

    The routine corrects for the given dispersion, calculates a
    background and corrects for the bandpass. Note that it does not
    flag any bad channels; these should have been flagged in the input
    data beforehand.

    When multiple polarization channels exist, it performs the above
    steps for each channel this for each independently, then combines
    the resulting dynamical spectra together into one dynamical
    spectrum.

    The data is either

    - two-dimensional, with the first dimension the time samples and
      the second dimension the frequency channels,

    - or three-dimensional, with the first dimension the time samples,
      the second dimension the polarization, and the third dimension
      the frequency channels.

    The data should contain either one or no polarization dimension;
    in the latter case this is assumed to be Stokes I. Or the data
    contains four polarization channels, of which the first two are
    assumed to be xx and yy and will be added together.

    If the dispersion `dm` is set to 0 or None, no dispersion
    correction is performed. If the `backgroundrange` is set to
    `None`, no background is calculated and no bandpass correction is
    performed.

    Parameters
    ----------
    data : Array or a dict of str, Array
        This is a single three-dimensional array, or a two-dimensional
        array, or a dictionary with keys of "xx" and "yy" and
        two-dimensional arrays (time and frequency) as values. For the
        three-dimensional array, the second dimension should be the
        "polarization" channel, in this case only the xx/yy data
        (i.e., of size 2).

    freqs: ndarray
        Frequencies corresponding to the data

    tsamp: float
        Sampling time in milliseconds

    dm: float
        Dispersion measure. Zero or `None` will skip the dedispersion step

    reffreq: float, option.
        Reference frequency for the dedispersion. If not given (or
        `None`), the maximum frequency from the `freqs` array is used.

    backgroundrange : iterable of 2-tuple of float, optional
        Fractional range along the time axis where the background
        should be measured. See `calc_background` for details.

        Set to `None` to not apply a bandpass correction (no background
        is calculated).

        The bandpass correction is calculated separately for each
        polarization (if applicable), but with identical
        `backgroundrange` s for each polarization channel.

    bkg_method : str, optional
        What method to use for the background calculation. One of
        "mean", "median" or "mode". See `calc_background` for details.

    combine : str
        Method to combine individual channels (if applicable). One of "mean",
        "average" (same as "mean") or "sum". Default is "mean".

    Returns
    -------
    Tuple of Array, (float, float)
        Tuple of
        - two-dimensional array with the Stokes intensity parameter.
        - background value and standard deviation

    Raises
    ------
    ValueError
        - if 'xx' does not exist as a key for the case where data is a dict.
        - if the length of `freqs` doesn't match the first axis of `data`.
        - for an invalid `combine` argument.

    """

    if combine not in ("mean", "average", "sum"):
        raise ValueError('`combine` is not one of "mean", "average" or "sum"')

    (tsamp, freqs, reffreq, dm, _) = ensure_quantities(tsamp, freqs, reffreq, dm, None)

    spectra = _create_dynspectra(
        data, freqs, tsamp, dm, reffreq, backgroundrange, bkg_method, background
    )
    xx_bkgmean = spectra["xx"][1]
    xx_bkgstd = spectra["xx"][2]
    yy_bkgmean = spectra["yy"][1]
    yy_bkgstd = spectra["yy"][2]
    if spectra["yy"][0] is None:
        spectrum = spectra["xx"][0]
    else:
        spectrum = spectra["xx"][0] + spectra["yy"][0]
        if combine != "sum":
            spectrum /= 2

    bkg_mean = xx_bkgmean + yy_bkgmean
    bkg_std = np.sqrt(xx_bkgstd**2 + yy_bkgstd**2)
    if spectra["yy"][0] is not None and combine != "sum":
        bkg_mean /= 2
        bkg_std /= np.sqrt(2)

    return spectrum, (bkg_mean, bkg_std)


def calc_lightcurve(
    data: dict[str, Array],
    freqs: list | np.ndarray | Quantity,
    tsamp: float | Quantity,
    dm: float | Quantity,
    reffreq: float | Quantity | None = None,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
    background: tuple[float | dict, float | dict] | None = None,
) -> tuple[Array, tuple[float, float]]:
    """Calculate the light curve by summing across channels, after
    dedispersion, flagging bad channels and background correction.

    This returns a one-dimensional array of summed intensity versus
    samples. Optionally, the average standard deviation is returned.

    The light curve is computed from the two-dimensional intensity
    array, and the arguments are identical to that of
    ``create_dynspectrum``.

    Parameters
    ----------

    data : dict[str, Array]
        array for value.

    freqs: ndarray
        Frequencies corresponding to the data

    tsamp: float
        Sampling time in milliseconds

    dm: float
        Dispersion measure. Zero or `None` will skip the dedispersion step

    reffreq: float, option.
        Reference frequency for the dedispersion. If not given (or
        `None`), the maximum frequency from the `freqs` array is used.

    backgroundrange: 2-tuple of background interval, or iterable of
        2-tuples of background fraction intervals.

        Each interval is a 2-tuple that contains two floating point
        values between 0 and 1, which are the fractions of the full
        data sample (time-axis) range that contain a background
        section. All sections are combined, after which the background
        is calculated (using the median or mean value over the
        combined area).

    bkg_method : str, optional
        What method to use for the background calculation. One of
        "mean", "median" or "mode". See `calc_background` for details.

    Returns
    -------
    A 2-tuple of (ndarray, (float, float))
        The one-dimensional light curve, plus the background and its standard deviation

    """

    results = create_dynspectrum(
        data,
        freqs,
        tsamp,
        dm,
        reffreq,
        backgroundrange,
        bkg_method=bkg_method,
    )

    results, bkg = results

    bkg_mean, bkg_std = bkg
    # The background for the intensity is only summed across
    # samples / time, we need to average this array of standard deviations
    # along the channels / frequency
    bkg_mean = np.mean(bkg_mean)
    bkg_std = np.sqrt(np.mean(bkg_std**2))

    lightcurve = results.sum(axis=1)

    return lightcurve, (bkg_mean, bkg_std)


def calc_lightcurve_from_waterfall(waterfall: Array) -> Array:
    """Calculate the light curve from waterfall data

    This is done by simply summing over the second (frequency) axis

    Parameters
    ----------
    waterfall : ndarray
        the dynamical spectrum

    Returns
    -------
    lightcurve : ndarray
        The one-dimensional light curve

    Raises
    ------
    ValueError
        If `waterfall` is not two dimensional

    """

    if waterfall.ndim != 2:
        raise ValueError("argument is not two-dimensional")

    lightcurve = waterfall.sum(axis=1)

    return lightcurve


def bowtie(
    data: Array,
    freqs: list | np.ndarray | Quantity,
    tsamp: float | Quantity,
    dminterval: FInterval | QInterval,
    reffreq: float | Quantity | None = None,
    ndm: int = 50,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Create the data for a bowtie plot: varying DM versus time/samples

    Parameters
    ----------
    data : np.ndarray
        data containing freq on the y-axis and time on the x-axis

    freqs : np.ndarray
        frequencies corresponding to the channel centers

    tsamp : float
        sampling time interval in seconds

    dminterval : tuple[float, float]
        range of the dispersion measure: start and stop

        An average DM is calculated from this range, which is then
        used in the calculation of the background: the data is
        dedispersed to this mean DM and the background is calculated,
        which is used for the bandpass correction.

    reffreq: float or None

        reference frequency used for dispersion. If None, use the
        highest value of the given `freqs`.

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
    tuple of [np.ndarray, np.ndarray]
        - first item: bowtie data
          two dimensional array containing the bowtie-plot data
        - second item: dm values matching the y-axis of the bowtie data

    Raises
    ------
    ValueError
        if the length of `freqs` doesn't match the first axis of `data`.

    """

    if data.shape[-1] != len(freqs):
        raise ValueError(
            "`freqs` length does not match the last axis of the data array"
        )

    data, poltype = _format_data(data)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError("data contains multiple polarization channels")

    (tsamp, freqs, reffreq, _, dminterval) = ensure_quantities(
        tsamp, freqs, reffreq, None, dminterval
    )

    dmcenter = (dminterval[0] + dminterval[1]) / 2
    # Dedisperse to the central frequency
    # This will also background-correct / normalize the spectrum
    spectra = _create_dynspectra(
        data, freqs, tsamp, dmcenter, reffreq, backgroundrange, bkg_method
    )

    data = spectra["xx"][0]  # There is only one channel

    # Dedisperse the corrected spectrum across dmrange;
    # dmrange is relative to dmcenter
    dms = np.linspace(dminterval[0], dminterval[1], ndm) - dmcenter
    tie = []
    for dm in dms:
        datadd = dedisperse(data, freqs, tsamp, dm)
        # Add all data into a light curve, taking care of NaNs
        datadd = np.ma.filled(datadd, np.nan)
        lc = np.nansum(datadd, axis=1)
        tie.append(lc)
    tie = np.vstack(tie)

    return tie, dms + dmcenter


def signal2noise(
    data: Array,
    freqs: list | np.ndarray | Quantity,
    tsamp: float | Quantity,
    dminterval: FInterval | QInterval,
    dm: float | Quantity | None = None,
    reffreq: float | None = None,
    ndm: int = 50,
    backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
    bkg_method: str = "mean",
    background: tuple[float | dict, float | dict] = None,
    peak: bool = True,
) -> tuple[Array, Array]:
    """Calculate peak signal to noise values over a range of DM

    This calculates the light curve (dynamical spectrum summed across
    the channels) for variying dispersion measures, then obtains the
    peak intensity for each DM


    Parameters
    ----------
    data : np.ndarray
        data containing freq on the y-axis and time on the x-axis

    freqs : np.ndarray
        frequencies corresponding to the channel centers

    tsamp : float
        sampling time interval in seconds

    dminterval : tuple[float, float]
        interval of the dispersion measure: start and stop

        An average DM is calculated from this range, which is then
        used in the calculation of the background: the data is
        dedispersed to this mean DM and the background is calculated,
        which is used for the bandpass correction.

        If `dm` is given, however, this is used as the zeropoint of
        the dispersion measure instead.

    dm : float, optional

        An optional specific dispersion measure to use in the
        background calculation and bandpass correction; if not given,
        taken from the mid of `dminterval`.

    reffreq: float or None

        reference frequency used for dispersion. If None, use the
        highest value of the given `freqs`.

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

    If the `background` argument is not `None`, `backgroundrange` and
    `bkg_method` are ignored. If `bkg_extra` is also set, the returned
    values identical to the given values.


    Returns
    -------
    Tuple of dm values and (peak) signal to noise ratios, both as a NumPy array.

    """

    if data.shape[-1] != len(freqs):
        raise ValueError(
            "`freqs` length does not match the last axis of the data array"
        )

    has_unit = isinstance(dminterval[0], Quantity)

    (tsamp, freqs, reffreq, dm, dminterval) = ensure_quantities(
        tsamp, freqs, reffreq, dm, dminterval
    )

    if dm is None:
        dmcenter = (dminterval[0] + dminterval[1]) / 2
    else:
        dmcenter = dm
    spectra = _create_dynspectra(
        data, freqs, tsamp, dmcenter, reffreq, backgroundrange, bkg_method
    )
    waterfall = spectra["xx"][0]
    lightcurve = np.ma.filled(waterfall, 0).sum(axis=1)
    idx_bkg = []
    nsamp = len(lightcurve)
    # Calculate the background of the light curve
    # using the `backgroundrange`
    if backgroundrange:
        for bkgrange in backgroundrange:
            low = int(nsamp * bkgrange[0] + 0.5)
            high = int(nsamp * bkgrange[1] + 0.5)
            idx_bkg.append(np.arange(low, high))
        idx_bkg = np.concatenate(idx_bkg)
    lcstd = lightcurve[idx_bkg].std()

    # dms is relative to the mean DM
    dms = np.linspace(dminterval[0], dminterval[1], ndm) - dmcenter

    logger.info(
        "Iterating over %d DMs from %.4f to %.4f", len(dms), dms[0].value, dms[-1].value
    )
    ratios = []
    for i, dm in enumerate(dms):
        logger.debug("dm = %.4f", dm.value)
        relwaterfall = dedisperse(waterfall, freqs, tsamp, dm, reffreq=reffreq)
        # Sum across frequencies to obtain the light curve
        lightcurve = np.ma.filled(relwaterfall, 0).sum(axis=1)
        value = lightcurve.max() if peak else lightcurve.sum()
        ratio = value / lcstd
        ratios.append(ratio)

    dms += dmcenter

    if has_unit:
        return dms, np.asarray(ratios)
    else:
        return dms.value, np.asarray(ratios)


def fit_ratios(dms, ratios) -> tuple[float, float, float]:
    """Perform a least-squares fit of a Gaussian curve to the
    signal-to-noise ratios

    Returns the amplitude, mean and standard deviation of the fitted
    curve

    Note that the input parameters are essentially the two values that
    `signal2noise` returns.

    Parameters
    ----------
    dms : list or np.ndarray
        list or array of dispersion measure values

    ratios : list or np.ndarray
        list or array of signal-to-noise ratios

    Returns
    -------
    3-tuple of floats
        The amplitude, mean and standard deviation of the fitted Gaussian

    """

    fitter = TRFLSQFitter()
    stddev = (max(dms) - min(dms)) / 4  # rough estimate
    model = Gaussian1D(amplitude=max(ratios), mean=np.median(dms), stddev=stddev)
    fit = fitter(model, dms, ratios)
    ampl = fit.amplitude.value
    # Handle potential quantities
    if fit.mean.unit:
        mean, stddev = fit.mean.quantity, fit.stddev.quantity
    else:
        mean, stddev = fit.mean.value, fit.stddev.value
    logger.info(
        "Ratio fit to Gaussian; result amplitude, mean +/- stddev = %.3f, %.3f +/- %.3f",
        ampl,
        mean,
        stddev,
    )

    return ampl, mean, stddev
