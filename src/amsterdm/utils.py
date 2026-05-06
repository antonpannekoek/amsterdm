"""Utility functions and classes"""

from typing import NamedTuple

from astropy.units import Quantity
import numpy as np


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
