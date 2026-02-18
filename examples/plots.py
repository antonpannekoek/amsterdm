#! /usr/bin/env python

"""Create one or more plots for a bust file

Usage example:

    python examples/plots.py examples/amsterdm-sim.fits --plots s2n waterfall lc --back 0 0.3333 --back 0.666 1 --dm 123.45


Run examples/simulate.py to create an input file for this example:

    python examples/simulate.py

"""

import logging
from pathlib import Path

import numpy as np

import amsterdm
import amsterdm.plot as dmplot
from amsterdm import core


logger = logging.getLogger("amsterdm")


def setup_logger(loglevel):
    fmt = "%(asctime)s  [%(levelname)-5s] - %(module)s.%(funcName)s():%(lineno)d: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(loglevel)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def main(
    path,
    dm,
    plots,
    background,
    badchannels=None,
    s2nrange=None,
    s2nsection=None,
    times=None,
    freqs=None,
    loglevel=logging.INFO,
):
    setup_logger(loglevel)
    logger.info("Reading file %s", path)
    if s2nrange:
        s2nrange = (s2nrange[0], s2nrange[1], int(round(s2nrange[2])))
    else:  # default to DM +/- 1, 50 steps
        s2nrange = (dm - 1, dm + 1, 50)
    with amsterdm.openfile(path) as burst:
        logger.info("Done reading file")
        pngfile = path.with_suffix(".png")
        if badchannels is None:
            badchannels = []
        if times or freqs:
            burst.trim(times=times, freqs=freqs)
        # badchannels = np.hstack([np.arange(0, 255, 32), np.arange(1, 255, 32), np.arange(30, 255, 32), np.arange(31, 255, 32)])
        # badchannels = [1, 9, 15, 16, 17, 76, 79, 80, 81, 82, 83, 98, 99, 100, 103, 111, 112, 113, 123, 127]
        if burst.header["foff"]:
            # Flip the bad channels
            nchan = burst.data.shape[-1]
            badchannels = [nchan - channel for channel in badchannels]

        if s2nsection:
            sections = [s2nsection]
        else:
            sections = []
        if (
            "lightcurve" in plots
            or "lc" in plots
            or "ratio" in plots
            or "s2n" in plots
            or "grid" in plots
        ) and not sections:
            logger.info("Creating light curve")
            lc = burst.lightcurve(dm, badchannels, backgroundrange=background)
            logger.info("Determining sections: %s", sections)
            sections = core.findrangelc(lc, kappa=10)

        if "all" in plots or "waterfall" in plots or "dynspec" in plots:
            fig, ax = dmplot.waterfall(
                burst,
                dm,
                badchannels,
                backgroundrange=background,
                fillmask=np.ma.median,
            )
            ax.set_title(f"Waterfall plot of {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-waterfall")
            fig.savefig(outfile)

        if "all" in plots or "bowtie" in plots:
            fig, ax = dmplot.bowtie(
                burst,
                (dm - 50, dm + 50),
                badchannels,
                backgroundrange=background,
                ndm=50,
            )
            ax.set_title(f"Bowtie plot of {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-bowtie")
            fig.savefig(outfile)

        if "all" in plots or "lightcurve" in plots or "lc" in plots:
            logger.info("Creating light curve plot")
            fig, ax = dmplot.lightcurve(
                burst, dm, badchannels, backgroundrange=background
            )
            if sections:
                vlines = [item for interval in sections for item in interval]
                ax.vlines(
                    vlines,
                    0,
                    1,
                    transform=ax.get_xaxis_transform(),
                    alpha=0.2,
                    color="k",
                )
            ax.set_title(f"Light curve of {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-lightcurve")
            fig.savefig(outfile)

        if "all" in plots or "background" in plots or "bg" in plots:
            fig, ax = dmplot.background(
                burst, dm, badchannels, backgroundrange=background
            )
            ax.set_title(f"Background statistics of {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-background")
            fig.savefig(outfile)

        if "all" in plots or "ratio" in plots or "s2n" in plots:
            section = None
            if sections:
                # combine sections into one big section
                section = [sections[0][0], sections[-1][1]]
                # # Extend it on both sides to cover dispersion ("rolling the axis")
                # section = (max(0, section[0]), min(nsamples, section[1]))

                # convert to fractions
                nsamples = burst.data.shape[0]
                section = (section[0] / nsamples, section[1] / nsamples)
                delta = section[1] - section[0]
                section = (
                    max(section[0] - delta / 2, 0),
                    min(section[1] + delta / 2, nsamples),
                )
            dms = np.linspace(*s2nrange)

            peak = True
            fig, ax = dmplot.signal2noise(
                burst,
                dms,
                badchannels=badchannels,
                backgroundrange=background,
                peak=peak,
                peak_interval=section,
                fit=True,
            )
            if peak:
                ax.set_title(f"Peak signal to noise for {path.stem}")
            else:
                ax.set_title(f"Signal to noise for {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-s2n")
            fig.savefig(outfile)

        if "all" in plots or "grid" in plots:
            section = None
            if sections:
                nsamples = burst.data.shape[0]
                # combine sections into one big section
                section = [sections[0][0], sections[-1][1]]
                # Extend it on both sides to cover dispersion ("rolling the axis")
                section = (max(0, section[0] - 1000), min(nsamples, section[1] + 1000))
                # convert to fractions
                section = (section[0] / nsamples, section[1] / nsamples)
            dms = np.linspace(*s2nrange)
            peak = True
            fig, ax = dmplot.grid(
                burst,
                dm=dm,
                dms=dms,
                badchannels=badchannels,
                backgroundrange=background,
                peak=peak,
                peak_interval=section,
                title=path.stem,
            )

            outfile = pngfile.with_stem(path.stem + "-grid")
            fig.savefig(outfile)


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("file", help="Filterbank file")
    parser.add_argument("--dm", type=float, default=219.356)
    parser.add_argument(
        "--plots",
        choices=(
            "waterfall",
            "dynspec",
            "bowtie",
            "lightcurve",
            "lc",
            "background",
            "bg",
            "ratio",
            "s2n",
            "all",
            "grid",
        ),
        default="all",
        nargs="+",
    )
    parser.add_argument(
        "--badchannels", type=int, nargs="+", help="list of channel numbers"
    )
    parser.add_argument(
        "--back",
        nargs=2,
        type=float,
        action="append",
        help="Set of start and end (time) fractions for the background estimate",
    )
    parser.add_argument(
        "--s2n-range",
        nargs=3,
        type=float,
        default=None,
        help="DM range for S/N calculation. Three values: low, high, number (2 floats + 1 int)",
    )
    parser.add_argument(
        "--s2n-section", nargs=2, type=float, help="Sample section to compute S/N for"
    )
    parser.add_argument(
        "--times",
        nargs=2,
        type=float,
        help="Time range to select data (in milliseconds)",
    )
    parser.add_argument(
        "--freqs", nargs=2, type=float, help="Frequency range to select data (in MHz)"
    )

    # parser.add_argument(
    #    '--plot-extremes',
    #    action='store_true',
    #    help=("If a 's2n' plot is selected (and thus a DM range exist), "
    #          "and a 'waterfall' or 'lightcurve' plot is also selected, "
    #          "plot the latter plot(s) at the extremes of the DM range")
    # )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        Path(args.file),
        args.dm,
        args.plots,
        args.back,
        badchannels=args.badchannels,
        s2nrange=args.s2n_range,
        s2nsection=args.s2n_section,
        times=args.times,
        freqs=args.freqs,
    )
