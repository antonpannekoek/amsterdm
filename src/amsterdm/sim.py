import numpy as np

from .constants import DMCONST


def disperse(
    data: np.ndarray,
    dm: float,
    reffreq: float,
    freqs: np.ndarray,
    tsamp: float,
    dmconst: float = DMCONST,
):
    # init empty array to store dedisp data in
    newdata = np.empty_like(data)

    # calculate time shifts and convert to bin shifts
    time_shift = -dmconst * dm * (reffreq**-2.0 - freqs**-2.0)

    # round to nearest integer
    bin_shift = np.rint((time_shift / tsamp)).astype(np.int64)

    print(data.shape, len(bin_shift))
    # checks
    assert len(bin_shift) == data.shape[0]

    # dedisperse by rolling the channels forward
    for i, bs in enumerate(bin_shift):
        newdata[i, :] = np.roll(data[i, :], bs)

    return newdata


def simulate(
    peaks,
    t0s,
    widths,
    dm,
    dmreffreq,
    nsamples,
    nchannels,
    freq0,
    dfreq,
    time0,
    tsamp,
    xy=None,
    bandpass=None,
    background=(5, 1),
    seed=None,
):
    nstokes = 4 if xy else 1
    if not xy:
        xy = [1]
    elif isinstance(xy, (int, float)):
        xy = [xy]
    if not isinstance(xy, (list, tuple)) or len(xy) not in [1, 2]:
        raise ValueError(
            "'xy' should be None, an single value, or a 1- or 2-element list or tuple"
        )

    channels = np.arange(nchannels)
    freqs = channels * dfreq + freq0
    times = np.arange(nsamples) * tsamp + time0
    samples = np.arange(nsamples)
    tend = time0 + nsamples * tsamp

    # For the moment, the bandpass is the same for each channel
    if bandpass is None:
        bandpass = np.ones(nchannels)
    elif isinstance(bandpass, (int, float)):
        bandpass = bandpass * np.ones_like(channels)
    elif callable(bandpass):
        bandpass = bandpass(channels)
    elif isinstance(bandpass, list) and len(bandpass) == nchannels:
        bandpass = np.asarray(bandpass)
    elif isinstance(bandpass, np.ndarray) and bandpass.shape == (nchannels,):
        pass
    else:
        raise ValueError("'bandpass' has an incorect type")

    rng = np.random.default_rng(seed=seed)
    dynspec = rng.normal(
        loc=background[0], scale=background[1], size=(nsamples, nstokes, nchannels)
    )
    # Apply the bandpass to the noise, across the channels dimension
    dynspec *= bandpass[None, None, :]

    fluxes = []  # individual light curves for each peak
    for i, (t0, peak, width) in enumerate(zip(t0s, peaks, widths)):
        if t0 < time0 or t0 > tend:
            raise ValueError("burst time outside of time range")
        # Convert the time parameters to sample space
        delta = width / tsamp
        sample0 = t0 / tsamp
        flux = peak * np.exp(-((samples - sample0) ** 2) / (2 * delta**2))
        fluxes.append(flux)

    # Add the individual bursts to the noise
    # Do this independently for each polarization channel,
    # and apply the bandpass
    for flux in fluxes:
        for i, fxy in enumerate(xy):
            dynspec[:, i, :] += flux[:, None] * fxy * bandpass[None, :]
        # The remaining two polarization channels will contain only noise

    for i in range(len(xy)):
        dynspec[:, i, :] = disperse(dynspec[:, i, :].T, dm, dmreffreq, freqs, tsamp).T

    # remove line below; only to silence the linter
    return dynspec, times, freqs
