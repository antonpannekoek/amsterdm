"""Module to read a HDF 5 file with FRB data

Implemented for CHIME waterfall data files

"""

import logging
from pathlib import Path

from astropy.time import Time
import h5py
import numpy as np


logger = logging.getLogger(__package__)


def read_hdf5(filename: str | Path) -> tuple[dict, np.ndarray]:
    h5file = h5py.File(filename)
    try:
        group = h5file["frb"]
    except KeyError:
        logger.error("No 'frb' group found in HDF 5 file")
        raise ValueError("No 'frb' group found in HDF 5 file")

    try:
        # convert from milliseconds
        times = group["plot_time"][:] / 1000
    except IndexError:
        logger.error("Missing 'plot_time' dataset")
        raise ValueError("Missing 'plot_time' dataset")
    try:
        freqs = group["plot_freq"][:]
    except IndexError:
        logger.error("Missing 'plot_freq' dataset")
        raise ValueError("Missing 'plot_freq' dataset")

    header = {
        "reltimes": times,
        "freqs": freqs,
    }
    header["dm"] = group.attrs["dm"]
    header["object"] = group.attrs["tns_name"]
    header["src_name"] = header["object"]
    header["tsamp"] = np.median(np.diff(times))

    header["fch1"] = freqs[0]
    header["foff"] = np.median(np.diff(freqs))
    header["fanchor"] = "mid"
    # Set some defaults
    # Arbitrary observation starting point
    header["tstart"] = Time("2001-01-01T00:00:00").mjd

    try:
        data = group["wfall"][...].T
    except IndexError:
        logger.error("Missing 'wfall' dataset")
        raise ValueError("Missing 'wfall' dataset")

    # Flag the NaN data as masked
    data = np.ma.masked_invalid(data)

    return header, data
