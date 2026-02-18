"""Module to read Filterbank data"""

import os
import struct

import numpy as np

from ..utils import coord2deg


# The following two values are valid when the endianness of the
# platform is given for struct packing & unpacking
INTSIZE = 4  # Number of bytes for an integer number
DOUBLESIZE = 8  # Number of bytes for a double precision floating point number


# All header keys with their data types
HEADER_KEYS = {
    "rawdatafile": "string",
    "source_name": "string",
    "src_raj": "double",
    "src_dej": "double",
    "az_start": "double",
    "za_start": "double",
    "tstart": "double",
    "tsamp": "double",
    "fch1": "double",
    "foff": "double",
    "machine_id": "int",
    "barycentric": "int",
    "pulsarcentric": "int",
    "telescope_id": "int",
    "data_type": "int",
    "nchans": "int",
    "nbeams": "int",
    "ibeam": "int",
    "nbits": "int",
    "nifs": "int",
}

# dtype dervied from the number of bits, nbits
DTYPES = {
    8: np.uint8,
    16: np.uint16,
    32: np.float32,
}


class Header(dict):
    """Simple dict-inherited class for the header; helper class"""


def read_string(file, le=True):
    """Read a string from a Filterbank header"""
    # Mind the endianness
    fmt = "<i" if le else ">i"
    # Get number of characters to read from the initial integer
    (n,) = struct.unpack(fmt, file.read(INTSIZE))
    string = file.read(n)
    return string.decode()


def read_int(file, le=True):
    """Read an integer nunmber from a Filterbank header"""
    # Mind the endianness
    fmt = "<i" if le else ">i"
    (n,) = struct.unpack(fmt, file.read(INTSIZE))
    return n


def read_double(file, le=True):
    """Read a double precision floating point number from a Filterbank header"""
    # Mind the endianness
    fmt = "<d" if le else ">d"
    (x,) = struct.unpack(fmt, file.read(DOUBLESIZE))
    return x


def read_header(file, le=True):
    """Read the header part of a Filterbank file"""
    header = Header()
    filepos = file.tell()
    file.seek(0)

    string = read_string(file)
    if string != "HEADER_START":
        raise ValueError("no Filterbank header found in file")

    while True:
        keyword = read_string(file)
        if keyword == "HEADER_END":
            header._file_info = {
                "start": filepos,
                "end": file.tell(),
                "size": file.tell() - filepos,
            }
            break
        htype = HEADER_KEYS.get(keyword)
        if not htype:
            raise ValueError(f"unknown header keyword {keyword}")
        if htype == "string":
            value = read_string(file)
        elif htype == "double":
            value = read_double(file)
        elif htype == "int":
            value = read_int(file)
        header[keyword] = value

    # Some additional header keywords
    if "bw" not in header:
        header["bw"] = header["nchans"] * header["foff"]
    if "cfreq" not in header:
        header["cfreq"] = header["fch1"] + header["bw"] / 2 - header["foff"] / 2
    header["poln_order"] = "I" if header["nifs"] == 1 else 4
    header["src_ra"] = coord2deg(header["src_raj"], 15)
    header["src_dec"] = coord2deg(header["src_dej"])

    return header


def read_data(file, header, le=True):
    nchan = header["nchans"]  # number of channels / frequencies
    nbits = header["nbits"]  # bits per value
    nifs = header["nifs"]  # number of polarisation channels
    nbytes = nbits // 8
    dtype = DTYPES.get(nbits)
    if not dtype:
        raise ValueError(f"{nbits} bits data type not supported")

    # data starts at the end of the header
    start = header._file_info["end"]
    # Get the data size from the file size minus the header size
    file.seek(0, os.SEEK_END)
    datasize = file.tell() - start
    # Read all data in one go
    # That should match the datasize, so assert this
    file.seek(start)
    data = file.read()
    assert datasize == len(data)
    bytes_spectrum = nbytes * nchan * nifs
    nsamp = datasize // bytes_spectrum
    data = np.frombuffer(data, dtype=dtype)
    data = data.reshape((-1, nifs, nchan))
    assert data.nbytes == datasize
    assert data.size == nsamp * nchan * nifs
    assert data.shape == (nsamp, nifs, nchan)
    return data


def read_filterbank(file, le=True):
    header = read_header(file, le=le)
    data = read_data(file, header, le=le)
    return header, data
