import astropy.io.fits as pyfits


def read_fits(file):
    hdulist = pyfits.open(file)
    for hdu in hdulist:
        if hdu.data is not None:
            header = hdulist[0].header
            data = hdulist[0].data
            break
    else:
        raise ValueError("No data found in FITS file")
    dictheader = {}
    for key in header:
        key = key.lower()
        if key in ["simple", "history", "comment"]:
            continue
        dictheader[key] = header[key]
    return header, data
