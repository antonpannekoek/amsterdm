#! /usr/bin/env python

import logging
from pathlib import Path

import numpy as np

import amsterdm
import amsterdm.plot as dmplot
from amsterdm import core
from amsterdm.utils import FInterval


logger = logging.getLogger("amsterdm")


def setup_logger(loglevel):
    fmt = "%(asctime)s  [%(levelname)-5s] - %(module)s.%(funcName)s():%(lineno)d: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(loglevel)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def main(path, dm, plots, background, badchannels=None, loglevel=logging.INFO):
    setup_logger(loglevel)
    logger.info("Reading file %s", path)
    with amsterdm.openfile(path) as candidate:
        logger.info("Done reading file")
        pngfile = path.with_suffix(".png")
        if badchannels is None:
            badchannels = []

        # badchannels = np.hstack([np.arange(0, 255, 32), np.arange(1, 255, 32), np.arange(30, 255, 32), np.arange(31, 255, 32)])
        badchannels = [
            1,
            9,
            15,
            16,
            17,
            76,
            79,
            80,
            81,
            82,
            83,
            98,
            99,
            100,
            103,
            111,
            112,
            113,
            123,
            127,
        ]
        if candidate.header["foff"]:
            # Flip the bad channels
            nchan = candidate.data.shape[-1]
            badchannels = [nchan - channel for channel in badchannels]

        # candidate.downsample(2)

        sections = []
        if (
            "lightcurve" in plots
            or "lc" in plots
            or "ratio" in plots
            or "s2n" in plots
            or "grid" in plots
        ):
            logger.info("Creating light curve")
            lc = candidate.lightcurve(
                dm, badchannels, backgroundrange=FInterval(0.4, 1)
            )
            sections, _ = core.findrangelc(lc, kappa=10)

        if "all" in plots or "waterfall" in plots or "dynspec" in plots:
            ax = dmplot.waterfall(
                candidate, dm, badchannels, backgroundrange=background
            )
            ax.set_title(f"Waterfall plot of {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-waterfall")
            ax.figure.savefig(outfile)

        if "all" in plots or "bowtie" in plots:
            ax = dmplot.bowtie(
                candidate,
                (dm - 50, dm + 50),
                badchannels,
                backgroundrange=background,
                ndm=50,
            )
            ax.set_title(f"Bowtie plot of {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-bowtie")
            ax.figure.savefig(outfile)

        if "all" in plots or "lightcurve" in plots or "lc" in plots:
            logger.info("Creating light curve plot")
            ax = dmplot.lightcurve(
                candidate, dm, badchannels, backgroundrange=background
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
            ax.figure.savefig(outfile)

        if "all" in plots or "background" in plots or "bg" in plots:
            ax = dmplot.background(
                candidate, dm, badchannels, backgroundrange=background
            )
            ax.set_title(f"Background statistics of {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-background")
            ax.figure.savefig(outfile)

        if "all" in plots or "ratio" in plots or "s2n" in plots:
            section = None
            if sections:
                nsamples = candidate.data.shape[0]
                # combine sections into one big section
                section = [sections[0][0], sections[-1][1]]
                # Extend it on both sides to cover dispersion ("rolling the axis")
                section = (max(0, section[0] - 1000), min(nsamples, section[1] + 1000))
                # convert to fractions
                section = (section[0] / nsamples, section[1] / nsamples)
            dms = np.linspace(dm - 0.15, dm + 0.15, 50)
            peak = True
            ax = dmplot.signal2noise(
                candidate,
                dms,
                badchannels=badchannels,
                backgroundrange=background,
                peak=peak,
                peak_interval=section,
            )
            if peak:
                ax.set_title(f"Peak signal to noise for {path.stem}")
            else:
                ax.set_title(f"Signal to noise for {path.stem}")
            outfile = pngfile.with_stem(path.stem + "-s2n")
            ax.figure.savefig(outfile)

        if "all" or "grid" in plots:
            if sections:
                nsamples = candidate.data.shape[0]
                # combine sections into one big section
                section = [sections[0][0], sections[-1][1]]
                # Extend it on both sides to cover dispersion ("rolling the axis")
                section = (max(0, section[0] - 1000), min(nsamples, section[1] + 1000))
                # convert to fractions
                section = (section[0] / nsamples, section[1] / nsamples)
            dms = np.linspace(dm - 0.1, dm + 0.1, 50)
            peak = True
            ax = dmplot.grid(
                candidate,
                dm=dm,
                dms=dms,
                badchannels=badchannels,
                backgroundrange=background,
                peak=peak,
                peak_interval=section,
                title=path.stem,
            )

            outfile = pngfile.with_stem(path.stem + "-grid")
            ax.figure.savefig(outfile)


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.file), args.dm, args.plots, args.back, badchannels=args.badchannels)
