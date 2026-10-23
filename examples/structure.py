"""Example to calculate a dispersion measure by optimizing the structure parameter nrom((d/dt I(t))^2), following the method described in Sutinjo et al, 2023, 10.3847/1538-4357/ace774

This calculates a (relative) dispersion measure from optimizing (maximizing) the structure parameter, including a related range and thus an estimate of the errors on the optimized dispersion measure.

"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from amsterdm.burst import Burst
from amsterdm.structure import Structure


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
    # Create a test structure with 1024 samples and 64 channels
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
    assert data.shape == (1024, 64)

    header = {
        "fch1": 1.1e3,
        "foff": 2,
        "tsamp": 2e-5,
    }
    # This will issue a user warning, that 'nchans' is not found in the header
    # (since the provided header is very minimal)
    # and 'chans' is automatically determined from the data
    burst = Burst(header, data)

    tie, dms = burst.dmplot(dminterval=[-1, 1], backgroundrange=[0.5, 1], ndm=150)
    assert tie.shape == (150, 1024)  # first axis DMs, second axis samples
    assert len(dms) == 150

    if False:
        # Optional write to file; these can be used as input for the SHRINE implementation
        # E.g., `python maximise_structure.py -l test -t 20 -d 0  -s`
        # sampling time in microseconds
        dt = int(header["tsamp"] * 1e6)
        np.save("test_DMs.npy", dms.value)
        np.save(f"test_I_{dt}us.npy", tie)

    struct = Structure(tie, dms)
    print(struct)

    # We can calculate a best kc value first, but don't have to;
    # `struct.calc()` does it if `kc` is `None`, which is the default.
    kc = struct.calc_kc()
    # There is random noise, but `kc` will likely around 310
    print(f"Best {kc = }")
    # Note: `kc` is also available via `struct.kc`

    # struct.calc() will now use the previous calculated `kc`.
    optdm, lowdm, highdm, mindm, maxdm = struct.calc()
    # lowdm and highdm are the errors on optdm,
    # while mindm and maxdm are the corresponding boundary values
    # I.e., mindm = optdm + lowdm and maxdm = optdm + highdm
    # Note that lowdm is negative!

    # All values are AstroPy variables, hence we require `.value` to extract the floating point value
    # The reason for this is that AstroPy will preserve the units, but printing out the values
    # will show the units for each value individually, instead of only once at the end as done below
    print(
        f"Found a structured optimized DM of {optdm.value:.5f} {lowdm.value:+.5f}/{highdm.value:+.5f}  pc / cm3"
    )
    print(f"DM range = {mindm.value} – {maxdm.value}  pc / cm3")

    # Create a combined plot of
    # - the dynamic spectrum
    # - DM-time plot
    # - spectrum plot with k_c overplotted
    # - structure parameter vs DM plot
    # - adjusted structure parameter vs DM plot
    # - DM uncertainty plot
    # - detrended noise plot
    # - relative detrended noise plot
    # - the structure value versus the DM
    # -
    filename = "structure.pdf"
    logger.info("Creating plots; writing to %s", filename)

    fig, ax = plt.subplot_mosaic(
        [
            ["results", "results"],
            ["dynspectrum", "dmtime"],
            ["spectrum", "structure"],
            ["adj_structure", "uncertainty"],
            ["rel_uncertainty", "detr_noise"],
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

    burst.dynspectrumplot(backgroundrange=[0.5, 1], ax=ax["dynspectrum"], cbar=False)
    ax["dynspectrum"].set_title("Dynamic spectrum")

    burst.dmtimeplot(
        dminterval=[-1, 1],
        backgroundrange=[0.5, 1],
        ndm=150,
        ax=ax["dmtime"],
        cbar=False,
    )
    ax["dmtime"].set_title("DM-time plot")

    struct.plot_spectrum(ax=ax["spectrum"])

    struct.plot_structure(ax=ax["structure"])

    # The adjusted structure is the structure plus the
    # error (calculated as the relative error multiplied by the structure)
    struct.plot_adjusted_structure(ax=ax["adj_structure"])

    struct.plot_uncertainty(ax=ax["uncertainty"])

    struct.plot_rel_uncertainty(ax=ax["rel_uncertainty"])

    struct.plot_detrended_noise(ax=ax["detr_noise"])

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
