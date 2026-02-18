"""
Module to read PSR FITS files

Modified from the Your package,
commit 2d202eed13434cce4015ceb1cc796981bdb30d25,
https://github.com/thepetabyteproject/your/blob/main/your/formats/psrfits.py

Originally from the Presto project: https://github.com/scottransom/presto/blob/master/python/presto/psrfits.py
  May 11, 2010, Patrick Lazarus, May 11, 2010
  Jul 4, 2016; Mar 25, 2021, Scott Ransom additions for 1-bit and 2-bit data reading

"""

import logging
from pathlib import Path

import astropy.io.fits as pyfits
from astropy.table import Table
import numpy as np


logger = logging.getLogger(__package__)


def unpack_1bit(data: np.ndarray) -> np.ndarray:
    """Unpack 1-bit data that has been read in as bytes.

    Args:
        data (np.ndarray): array of unsigned 2-bit ints packed into an
        array of bytes.

    Returns:
        np.ndarray: unpacked array. The size of this array will
        be eight times the size of the input data.

    """
    b0 = np.bitwise_and(data >> 0x07, 0x01)
    b1 = np.bitwise_and(data >> 0x06, 0x01)
    b2 = np.bitwise_and(data >> 0x05, 0x01)
    b3 = np.bitwise_and(data >> 0x04, 0x01)
    b4 = np.bitwise_and(data >> 0x03, 0x01)
    b5 = np.bitwise_and(data >> 0x02, 0x01)
    b6 = np.bitwise_and(data >> 0x01, 0x01)
    b7 = np.bitwise_and(data, 0x01)
    return np.dstack([b0, b1, b2, b3, b4, b5, b6, b7]).flatten()


def unpack_2bit(data: np.ndarray) -> np.ndarray:
    """
    Unpack 2-bit data that has been read in as bytes.

    Args:
        data (np.ndarray): array of unsigned 2-bit ints packed into an array of bytes.

    Returns:
        np.ndarray: unpacked array. The size of this array will be four times the size of the input data.

    """
    piece0 = np.bitwise_and(data >> 0x06, 0x03)
    piece1 = np.bitwise_and(data >> 0x04, 0x03)
    piece2 = np.bitwise_and(data >> 0x02, 0x03)
    piece3 = np.bitwise_and(data, 0x03)
    return np.dstack([piece0, piece1, piece2, piece3]).flatten()


def unpack_4bit(data: np.ndarray) -> np.ndarray:
    """
    Unpack 4-bit data that has been read in as bytes.

    Args:
        data (np.ndarray): array of unsigned 4-bit ints packed into an array of bytes.

    Returns:
        np.ndarray: unpacked array. The size of this array will be twice the size of the input data.

    """
    piece0 = np.bitwise_and(data >> 0x04, 0x0F)
    piece1 = np.bitwise_and(data, 0x0F)
    return np.dstack([piece0, piece1]).flatten()


def read_subint(
    table: Table,
    header: pyfits.Header,
    subint: int = 0,
    apply_weights: bool = True,
    apply_scales: bool = True,
    apply_offsets: bool = True,
    pol: int = 0,
    npol: int = 1,
):
    """
    Read a PSRFITS subint from a open pyfits file object.
    Applys scales, weights, and offsets to the data.

    Args:
        apply_weights (bool): If True, apply weights. (Default: apply weights)
        apply_scales (bool): If True, apply scales. (Default: apply scales)
        apply_offsets (bool): If True, apply offsets. (Default: apply offsets)
        pol (int): which polarization to chose
        npol (int): number of polarizations to return

    Returns:
        np.ndarray: Subint data with scales, weights, and offsets applied in float32 dtype with shape (nsamps,nchan).

    """

    sdata = table["DATA"][subint]
    shape = sdata.squeeze().shape

    nsamp = header["nbin"]
    nchan = header["nchan"]
    nbits = header["nbits"]
    polorder = header["pol_type"]

    if nbits < 8:  # Unpack the bytes data
        if len(shape) == 2:
            if shape != (nsamp, nchan * nbits // 8):
                sdata = sdata.reshape(nsamp, nchan * nbits // 8)
        if nbits == 4:
            data = unpack_4bit(sdata)
        elif nbits == 2:
            data = unpack_2bit(sdata)
        elif nbits == 1:
            data = unpack_1bit(sdata)
        else:
            raise ValueError("nbits should be 8, 4, 2 or 1")

    elif npol == 4:
        data = sdata.squeeze().astype(float)
        shape = data.shape
        if len(shape) == 3:
            if shape[0] == 4:  # GBT data: roll last (time) axis to front
                data = np.moveaxis(data, 2, 0)
    else:
        # Handle 4-npol GUPPI/PUPPI data
        if len(shape) == 3 and shape[1] == header["npol"] and polorder == "AABBCRCI":
            logger.info("Polarization is AABBCRCI, summing AA and BB")
            data = np.zeros((nsamp, nchan), dtype=np.float32)
            data += sdata[:, 0, :].squeeze()
            data += sdata[:, 1, :].squeeze()
            data *= 0.5

        elif len(shape) == 3 and shape[1] == npol and polorder == "IQUV":
            logger.info("Polarization is IQUV")
            data = np.zeros((nsamp, nchan), dtype=np.float32)
            if pol == 0:
                logger.info("Reading only Stokes I")
                data += sdata[:, 0, :].squeeze()
            elif pol == 1:
                logger.info("Calculating right circular polarisation data")
                data = data + ((sdata[:, 0, :] + sdata[:, 3, :]) / 2).squeeze()
            elif pol == 2:
                logger.info("Calculating left circular polarisation data")
                data = data + ((sdata[:, 0, :] - sdata[:, 3, :]) / 2).squeeze()
            elif pol == 3:
                logger.info("Calculating horizontal linear polarisation data")
                data = data + ((sdata[:, 0, :] + sdata[:, 1, :]) / 2).squeeze()
            elif pol == 4:
                logger.info("Calculating vertical linear polarisation data")
                data = data + ((sdata[:, 0, :] - sdata[:, 1, :]) / 2).squeeze()
            else:
                raise ValueError(f"pol={pol} value not supported")

        elif len(shape) == 3 and shape[1] == npol and polorder == "AABB":
            logger.info("Polarization is AABB, summing AA and BB")
            data = np.zeros((nsamp, nchan), dtype=np.float32)
            data += sdata[:, 0, :].squeeze()
            data += sdata[:, 1, :].squeeze()
            data *= 0.5

        elif len(shape) == 4 and shape[-1] == 2 and polorder == "IQUV":
            logger.info(
                "Data is packed as two uint8 arrays. Concatenating them to get uint16"
            )
            logger.info("Polarization is IQUV. Reading only Stokes I")
            data = np.zeros((nsamp, nchan), dtype=np.float32)
            data1 = sdata[:, 0, :, 0].astype(np.uint16)
            data2 = sdata[:, 0, :, 1].astype(np.uint16)
            data += np.left_shift(data2, 8) + data1

        else:
            data = np.asarray(sdata)

    data = data.reshape((nsamp, npol, nchan)).astype(np.float64)
    if apply_scales:
        scales = table["DAT_SCL"][subint].reshape(npol, -1)
        data *= scales
    if apply_offsets:
        data += table["DAT_OFFS"][subint].reshape(npol, -1)
    if apply_weights:
        data *= table["DAT_WTS"][subint]

    freqs = table["DAT_FREQ"][0]

    return data, freqs


def read_psrfits(
    file: str | Path, memmap: bool = False
) -> tuple[pyfits.Header, np.ndarray]:
    hdulist = pyfits.open(file, memmap=memmap)
    hdu = hdulist[0]
    header = {}
    for key in hdu.header:
        key = key.lower()
        if key in ["simple", "history", "comment"]:
            continue
        header[key] = hdu.header[key]
    # Add information from other parts of the file to the header
    try:
        hdu = hdulist["SUBINT"]
    except KeyError:
        raise ValueError("Missing 'SUBINT' extension in FITS File")
    keys = [
        "pol_type",
        "npol",
        "reffreq",
        "chan_bw",
        "dm",
        "tbin",
        "nbits",
        "nbin",
        "nchan",
    ]
    for key in keys:
        try:
            header[key] = hdu.header[key]
        except KeyError:
            pass
    header["nsubints"] = hdu.header["naxis2"]

    table = Table(hdu.data)

    data, freqs = read_subint(table, header, npol=header["npol"])
    header["freqs"] = freqs

    if len(freqs) != data.shape[-1]:
        raise ValueError("I/O error: frequencies don't match data shape")

    return header, data
