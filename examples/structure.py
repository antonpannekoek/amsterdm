"""Example to calculate a dispersion measure by optimizing the structure parameter nrom((d/dt I(t))^2), following the method described in Sutinjo et al, 2023, 10.3847/1538-4357/ace774

This calculates a (relative) dispersion measure from optimizing (maximizing) the structure parameter, including a related range and thus an estimate of the errors on the optimized dispersion measure.

"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from amsterdm.burst import Burst
from amsterdm import structure


logger = logging.getLogger("amsterdm")


def create_test_structure(
    nsamples: int,
    nchannels: int,
    noise: float = 1,
    seed: None | int | np.random.Generator = None,
    bandpass=None,
    xc: list[float] | float | None = None,
    yc: list[float] | float | None = None,
    xs: list[float] | float = 1,
    ys: list[float] | float = 1,
    ampl: list[float] | float = 1,
):
    """Helper function to simulate some Gaussian pulses in two-dimensional FRB data"""

    rng = np.random.default_rng(seed)

    xx = rng.normal(scale=noise, size=(nchannels, nsamples))
    yy = rng.normal(scale=noise, size=(nchannels, nsamples))
    data = np.sqrt(xx**2 + yy**2)
    # data = np.zeros((nchannels, nsamples))
    noise = data.copy()

    yy, xx = np.indices(data.shape)

    signal = np.zeros(data.shape)
    if xc is not None and yc is not None:
        if (
            isinstance(xc, float)
            and isinstance(yc, float)
            and isinstance(xs, float)
            and isinstance(ys, float)
            and isinstance(ampl, float)
        ):
            gaussian = ampl * np.exp(
                -((xx - xc) ** 2 / (2 * xs**2) + (yy - yc) ** 2 / (2 * ys**2))
            )
            signal += gaussian
        elif (
            isinstance(xc, (list, tuple))
            and isinstance(yc, (list, tuple))
            and isinstance(xs, (list, tuple))
            and isinstance(ys, (list, tuple))
            and isinstance(ampl, (list, tuple))
        ):
            for xc_, yc_, xs_, ys_, ampl_ in zip(xc, yc, xs, ys, ampl):
                gaussian = ampl_ * np.exp(
                    -((xx - xc_) ** 2 / (2 * xs_**2) + (yy - yc_) ** 2 / (2 * ys_**2))
                )
                signal += gaussian
    data += signal

    # Add a bandpass effect if applicable
    if callable(bandpass):
        x = np.arange(0, data.shape[0])
        sens = bandpass(x)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(x, sens)
        fig.savefig("kernel.png")
        data *= sens[..., None]
        signal *= sens[..., None]

    # Ensure axes order is time/samples, channel/freqs
    return data.T, signal.T, noise.T


def bandpass(x):
    # Offset and half sine curve as bandpass
    return np.ones_like(x) + 2 * np.sin(x * np.pi / len(x))


def run():
    data, ideal, noise = create_test_structure(
        1024,
        64,
        noise=0.2,
        bandpass=bandpass,
        xc=[200, 230],
        yc=[36, 18],
        xs=[2, 2],
        ys=[12, 12],
        ampl=[2, 1.5],
    )

    header = {
        "fch1": 1.1e3,
        "foff": 2,
        "tsamp": 2e-5,
    }
    burst = Burst(header, data)

    tie, dms = burst.bowtie(dminterval=[-1, 1], backgroundrange=[0.5, 1], ndm=150)
    # Writing to file; these can be used as input for the SHRINE implementation
    # E.g., `python maximise_structure.py -l test -t 20 -d 0  -s`
    dt = int(header["tsamp"] * 1e6)
    np.save("test_DMs.npy", dms.value)
    np.save(f"test_I_{dt}us.npy", tie)

    struct = structure.Structure(tie, dms)
    # We can calculate a best kc value first, but don't have to;
    # `struct.calc()` does it if `kc` is `None`, which is the default.
    kc = struct.calc_kc()
    print(f"Best {kc = }")

    # struct.calc() will now simply use the previous calculated `kc`.
    optdm, lowdm, highdm, mindm, maxdm = struct.calc()
    print(
        f"Found a structured optimized DM of {optdm.value:.5f} {lowdm.value:+.5f}/{highdm.value:+.5f}  pc / cm3"
    )
    print(f"DM range = {mindm.value} – {maxdm.value}  pc / cm3")

    filename = "structure.png"
    logger.info("Creating plots; writing to %s", filename)

    fig, ax = plt.subplot_mosaic(
        [
            ["results", "results"],
            ["waterfall", "bowtie"],
            ["spectrum", "structure"],
            ["adj_structure", "uncertainty"],
            ["detr_noise", "reldetr_noise"],
        ],
        height_ratios=[1, 3, 3, 3, 3],
        figsize=(12, 16),
    )

    text = f"DM = {optdm.value:.5f} {lowdm.value:+.5f}/{highdm.value:+.5f}  pc / cm3"
    suptext = f"(range: {mindm.value:.5f} – {maxdm.value:.5f})"
    ax["results"].text(
        x=0.5, y=0.8, s=text, va="center", ha="center", fontsize=14, color="black"
    )
    ax["results"].text(
        x=0.5, y=0.4, s=suptext, va="center", ha="center", fontsize=12, color="black"
    )
    ax["results"].set_axis_off()

    burst.waterfall(backgroundrange=[0.5, 1], ax=ax["waterfall"], cbar=False)
    ax["waterfall"].set_title("Dynamic spectrum")

    burst.bowtieplot(
        dminterval=[-1, 1],
        backgroundrange=[0.5, 1],
        ndm=150,
        ax=ax["bowtie"],
        cbar=False,
    )
    ax["bowtie"].set_title("Bow-tie plot")

    struct.plot_spectrum(ax=ax["spectrum"])

    struct.plot_structure(ax=ax["structure"])

    struct.plot_adjusted_structure(ax=ax["adj_structure"])

    struct.plot_uncertainty(ax=ax["uncertainty"])

    struct.plot_detrended_noise(ax=ax["detr_noise"])

    struct.plot_relative_detrended_noise(ax=ax["reldetr_noise"])

    fig.suptitle("Structure diagnostic plots", y=0.92, fontsize=22)

    # Make some space for the axis labels
    # Otherwise, the labels overlap between subplots
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    fig.savefig(filename)


def main():
    # Change the logging level
    logging.basicConfig(level="INFO")

    run()


if __name__ == "__main__":
    main()
