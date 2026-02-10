import astropy.io.fits as pyfits


def read_fileformat(fobj):
    """Returns the file type of the file object `fobj`

    Reads the first 80 bytes from a (binary) file object to determine
    the file signature (the "magic number"). Recognized format are
    Filterbank, FITS (assumed PSRFITS) and HDF 5 (e.g. CHIME
    waterfall).

    Returns a string with one of "filterbank-le", "filterbank-be"
    (Filterbank little- and big-endian), "psrfits" or "hdf5" (all
    lowercase).

    """

    filepos = fobj.tell()
    fobj.seek(0)
    testbytes = fobj.read(80)
    # 30 is only "header_start" and "header_end"
    if len(testbytes) < 30:
        raise ValueError("file truncated")
    if testbytes[:16] == b"\x0c\x00\x00\x00HEADER_START":
        format = "filterbank-le"
    elif testbytes[:16] == b"\x00\x00\x00\x0cHEADER_START":
        format = "filterbank-be"
    elif testbytes[:30] == b"SIMPLE  =                    T":
        # Read the FITS header, check for relevant keywords
        fobj.seek(0)
        header = pyfits.getheader(fobj, ext=0)
        if header.get("fitstype") != "PSRFITS":
            raise ValueError(
                "FITS file does not appear to be PSRFITS: "
                'incorrect or missing "FITSTYPE" keyword'
            )
    elif testbytes[:8] == b"\x89HDF\x0d\x0a\x1a\x0a":
        format = "hdf5"  # e.g. CHIME waterfall data
    else:
        raise ValueError("unknown file format")
    fobj.seek(filepos)
    return format
