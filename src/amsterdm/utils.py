"""Utility functions and classes"""

import logging
from typing import NamedTuple
import warnings

import astropy.io.fits as pyfits
from astropy.units import Quantity
import numpy as np

from .constants import Array, HEADER_POLKEYS


logger = logging.getLogger(__package__)


__all__ = ["FInterval", "QInterval", "mask2d", "coord2deg", "symlog", "symlog10"]


class FInterval(NamedTuple):
    """Simple tuple to hold a floating point interval

    Note that there is no restriction that the two values are not the
    same and start is smaller than end.

    """

    start: float
    end: float


class QInterval(NamedTuple):
    """Simple tuple to hold an interval of two quantities

    Note that there is no restriction that the two quantities are not the
    same and start is smaller than end.

    """

    start: Quantity
    end: Quantity


def ensure_interval_list(data):
    """Ensures `data` is a list of intervals, even if `data` is a single interval"""

    if isinstance(data, (FInterval, QInterval)):
        return [data]
    if isinstance(data, (tuple, list)) and not isinstance(
        data[0], (FInterval, QInterval, tuple, list)
    ):
        return [data]
    # Try to convert to list
    # tuples, sets, iterators and lists of intervals will all be lists
    return list(data)


def mask2d(data, rowids):
    """Mask rows in a two-dimensional array"""

    data = np.ma.masked_array(data, mask=False)
    data[:, rowids] = np.ma.masked
    return data


def coord2deg(coord, factor=1):
    """Convert a Filterbank coordinate format to degrees"""
    sign = -1 if coord < 0 else 1
    degrees, coord = divmod(coord, 10_000)
    minutes, seconds = divmod(coord, 100)
    degrees = degrees + minutes / 60 + seconds / 3600
    return sign * degrees * factor


def symlog(data: np.ndarray) -> np.ndarray:
    """Convert `data` to a symmetric log, with negative values
    converted to negative log

    Converts `-x to -log(x)`.

    `symlog` works like Matplotlib's "symlog" scale, and similarly, it
    should be used mainly for visualization, not for real data
    calculations.

    Returns
    -------
        The symlog value of the input data.

    References
    ----------
        https://doi.org/10.1088%2F0957-0233%2F24%2F2%2F027001
    """

    data = np.sign(data) * np.log(1 + np.abs(data))

    return data


def symlog10(data: np.ndarray) -> np.ndarray:
    """Convert `data` to a symmetric log10, with negative values
    converted to -log10

    Converts `-x to -log(x)`.

    `symlog10` works like Matplotlib's "symlog" scale, and similarly, it
    should be used mainly for visualization, not for real data
    calculations.

    Returns
    -------
        The symlog value of the input data.

    References
    ----------
        https://doi.org/10.1088%2F0957-0233%2F24%2F2%2F027001

    """

    c = 1 / np.log(10)
    data = np.sign(data) * np.log10(1 + np.abs(data / c))

    return data


def _format_data(
    data: Array | dict[str, Array],
    header: dict | pyfits.Header = None,
    poltype: str = "",
    convert: bool = True,
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

    Alternatively, the `header` can also contain the polarization
    type, through a header keyword given in
    `constants.HEADER_POLKEYS`. The `poltype` argument takes
    precedence over the header, if both are given; a warning will be
    issued in case of an inconsistency between the two).

    The input can be a two-dimensional array, a three-dimensional
    array or a dict of various polarization channels: each dict value
    is a two-dimensional array for that specific channel. The keys
    indicate the polarization type of the input, and can be matching
    keys "i", "q", "u", "v", "xx", "yy", "aa", "bb", "cr", "ci"; all
    format strings and keys are case independent. There is a minimal
    check that if "xx" is present, "yy" should be present (and vice
    versa), and the same for "aa", "bb", "cr" and "ci"; for "q", "u"
    and "v", there should at least be a "i" key present as well. From
    the keys, the polarization type is deduced, and if `poltype` is
    not empty, a check is made to see if this matches.

    If the input is three-dimensional array, the deduced polarization
    type depends on the size of the polarization channel dimension
    (the second, center, axis): if it's 1, Stokes I is assumed; if 4,
    Stokes I, Q, U and V is assumed. 2 channels is not supported; pass
    these as a dict with "xx" and "yy" keys is wanted.

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

    if isinstance(
        header, dict
    ):  # Convert to pyfits.Header, which covers more use cases
        header = pyfits.Header(header)

    _poltype = _get_poltype(data, header=header, poltype=poltype)

    newdata = data
    if convert:
        newdata, _poltype = _convert_poldata(newdata, _poltype)

    return newdata, _poltype


def _get_poltype(data: Array | dict, header: pyfits.Header = None, poltype: str = ""):

    # Obtain an initial polarization type from the header, if any
    _poltype = hpoltype = None
    if header:
        for key in HEADER_POLKEYS:
            hpoltype = header.get(key)
            if hpoltype:
                hpoltype = hpoltype.lower()
                break
        if hpoltype and poltype != hpoltype:
            warnings.warn("header polarization type overridden by `poltype` argument")
            _poltype = poltype
        else:
            _poltype = hpoltype

    # List of valid `poltype`s and their matching dict keys
    polkeys = {
        "xy": ("xx", "yy", "xy", "yx"),
        "ab": ("aa", "bb", "cr", "ci"),
        "i": ("i",),
        "q": ("q",),
        "u": ("u",),
        "v": ("v",),
        "iquv": ("i", "q", "u", "v"),
        # "iq": ("i", "q"),
        # "iu": ("i", "u"),
        # "iv": ("i", "v"),
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
            ("yy", "yx"),
            ("yy", "xy"),
            ("aa", "bb"),
            ("aa", "cr"),
            ("aa", "ci"),
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
            if set(polkeys[poltype]) != set(keys):
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
        inv = {tuple(sorted(value)): key for key, value in polkeys.items()}
        # Note: polarization channels are in alphabetical order in polkeys
        skeys = tuple(sorted(keys))
        _poltype = inv[skeys]

    else:  # "Single multi-dimensional NumPy array
        if not isinstance(data, (np.ndarray, np.ma.MaskedArray)):
            raise ValueError("data is not an array")
        if data.ndim not in (2, 3):
            raise ValueError("data is not two- or three-dimensional")
        shape = data.shape

        # User overridden polarization type
        if poltype:
            _poltype = poltype.lower()
            if (
                _poltype in ("i", "q", "u", "v")
                and data.ndim == 3
                and data.shape[1] != 1
            ):
                logger.warning(
                    "setting polarization type to single Stokes parameter, but more polarization channels available"
                )
            if _poltype not in ("i", "q", "u", "v") and data.ndim == 2:
                logger.warning(
                    "only one polarization channel available, but the selected polarization type is not a single Stokes parameter"
                )
        else:
            # Determine the polarization from the data; verify with header type if available
            if data.ndim == 2:
                # Ignore the header keyword; assume Stokes I
                logger.info("Two dimensional data found; Stokes I assumed")
                _poltype = "i"
            elif data.ndim == 3:
                if data.shape[1] == 1:
                    if hpoltype and hpoltype != "i":
                        logger.warning(
                            "header polarization keyword does not match data shape"
                        )
                    logger.info(
                        "Three dimensional data found, with second dimension 1; Stokes I assumed"
                    )
                    _poltype = "i"
                elif data.shape[1] == 4:
                    # Use header to determine the polarization channels
                    _poltype = hpoltype
                    if not _poltype:
                        raise ValueError(
                            "polarization type can't be determined; please specify using the 'poltype' keyword"
                        )
                elif data.shape[1] == 3:
                    if hpoltype and hpoltype != "iqu":
                        warnings.wan(
                            "header polarization keyword does not match data shape"
                        )
                    logger.info(
                        "Three dimension data found, with second dimension 3: Stokes IQU assumed"
                    )
                    _poltype = "iqu"
                else:
                    raise ValueError("incorrect number of polarization channels")
            else:
                raise ValueError(f"invalid data dimensions: {data.shape}")

    # Transform to AmsterDM usage
    if _poltype:
        _poltype = _poltype.lower()
    if _poltype == "aa+bb":
        _poltype = "i"
    if _poltype == "xx-yy":
        _poltype = "xy"
    if _poltype in ("aa-bb", "aabbcrci"):
        _poltype = "ab"
    if _poltype == "rr-ll":
        _poltype = "rl"
    if _poltype == "aabb":
        raise ValueError("two-channel AABB polarization type not supported")

    if _poltype not in ["xy", "ab", "rl", "i", "iquv", "iqu"]:
        raise ValueError(f"unknown polarization type {_poltype}")

    return _poltype


def _convert_poldata(data: Array | dict, poltype: str):
    """Convert data to Stokes I(QUV), given the polarisation type"""

    if isinstance(data, dict):
        # List of valid `poltype`s and their matching dict keys
        polkeys = {
            "xy": ("xx", "yy", "xy", "yx"),
            "ab": ("aa", "bb", "cr", "ci"),
            "i": ("i",),
            "q": ("q",),
            "u": ("u",),
            "v": ("v",),
            "iquv": ("i", "q", "u", "v"),
            # "iq": ("i", "q"),
            # "iu": ("i", "u"),
            # "iv": ("i", "v"),
            "iqu": ("i", "q", "u"),
        }
        keys = [key.lower() for key in data.keys()]
        skeys = tuple(sorted(keys))
        ndim = len(skeys)
        shape = data[skeys[0]].shape
        shape = (shape[0], ndim, shape[1])
        # Can we avoid all the copying here?
        newdata = np.empty(shape)
        for i, key in enumerate(polkeys[poltype]):
            newdata[:, i, :] = data[key][...]
    else:
        newdata = data.copy()

    if newdata.ndim == 2:
        if poltype not in ("i", "q", "u", "v"):
            raise ValueError("incorrect polarization type for two-dimensional data")
        return np.expand_dims(newdata, 1), poltype
    if newdata.ndim == 3 and newdata.shape[1] == 1:
        if poltype not in ("i", "q", "u", "v"):
            raise ValueError("incorrect polarization type for two-dimensional data")
        return newdata, poltype
    if poltype == "iquv":
        return newdata, poltype
    # Convert to Stokes IQUV
    if poltype == "xy":
        logger.info("Converting XX-YY data to IQUV")
        newdata = np.ma.array(
            [
                newdata[:, 0, :] + newdata[:, 1, :],
                newdata[:, 0, :] - newdata[:, 1, :],
                newdata[:, 2, :] + newdata[:, 3, :],
                newdata[:, 2, :] - newdata[:, 3, :],
            ]
        )
        newdata = np.moveaxis(newdata, 0, 1)
        poltype = "iquv"
    if poltype == "rl":
        logger.info("Converting RR-LL data to IQUV")
        newdata = np.ma.array(
            [
                newdata[:, 0, :] + newdata[:, 1, :],
                newdata[:, 2, :] + newdata[:, 3, :],
                newdata[:, 2, :] - newdata[:, 3, :],
                newdata[:, 0, :] - newdata[:, 1, :],
            ]
        )
        newdata = np.moveaxis(newdata, 0, 1)
        poltype = "iquv"
    if poltype == "ab":
        # AABBCRCI convention
        logger.info("Converting AA-BB data to IQUV")
        newdata = np.ma.array(
            [
                newdata[:, 0, :] + newdata[:, 1, :],
                newdata[:, 0, :] - newdata[:, 1, :],
                2 * newdata[:, 2, :],
                2 * newdata[:, 3, :],
            ]
        )
        newdata = np.moveaxis(newdata, 0, 1)
        poltype = "iquv"

    return newdata, poltype
