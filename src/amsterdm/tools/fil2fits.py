import argparse
from pathlib import Path

import astropy.io.fits as pyfits

from amsterdm.burst import openfile


def run(
    infile: str | Path,
    outfile: str | Path,
    duplicate_keys: bool = True,
    remove_keys: bool = False,
    overwrite: bool = False,
):
    with openfile(infile) as burst:
        header = burst.header

        if duplicate_keys:
            if "source_name" in header:
                header["srcname"] = header["source_name"]
                header["object"] = header["source_name"]
                if "fch1" in header:
                    header["fchan1"] = header["fch1"]
        if "src_ra" in header:
            header["ra"] = header["src_ra"]
        if "src_dec" in header:
            header["dec"] = header["src_dec"]

        if remove_keys:
            if "nchans" in header and header["nchans"] == burst.data.shape[2]:
                del header["nchans"]
            header.pop("src_raj", None)
            header.pop("src_dej", None)

        if "fanchor" not in header:
            header["fanchor"] = "mid"  # Assume the simplest default
        header = pyfits.Header(header)

        hdu = pyfits.PrimaryHDU(header=header, data=burst.data)
        hdu.writeto(outfile, overwrite=overwrite)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file name")
    parser.add_argument("output", nargs="?", help="Output file name")
    parser.add_argument(
        "--no-duplicate-keywords",
        action="store_false",
        dest="duplicate_keys",
        help="Do not duplicate values to more FITS-standard compliant keywords",
    )
    parser.add_argument(
        "--remove-keywords",
        action="store_true",
        help="Remove unnecessary or non FITS-standard keywords",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing output file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    infile = Path(args.input)
    outfile = args.output or infile.with_suffix(".fits")
    run(infile, outfile, args.duplicate_keys, args.remove_keywords, args.overwrite)


if __name__ == "__main__":
    main()
