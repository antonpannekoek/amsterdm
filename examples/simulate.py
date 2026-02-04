import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import amsterdm


peakfluxes = [5, 8, 4, 2]
t0s = [0.1, 0.15, 0.18, 0.22]  # in seconds
widths = [1e-2] * 4  # peak width in seconds

dm = 123.45
dmreffreq = 1400
nsamples = 20000
nchannels = 128
time0 = 0
timestep = 64e-6  # 64 microsec resolution
freq0 = 1400
dfreq = -1.6
xy = [1, 0.5]
# bandpass = np.sin(np.linspace(0, np.pi, nchannels))
bandpass = 10  # constant bandpass of 100
background = (5, 1)  # value and noise of background

simdata, times, freqs = amsterdm.sim.simulate(
    peakfluxes,
    t0s,
    widths,
    dm,
    dmreffreq,
    nsamples,
    nchannels,
    freq0,
    dfreq,
    time0,
    timestep,
    xy=xy,
    bandpass=bandpass,
    background=background,
)

header = pyfits.Header()
header["object"] = ("N/A", "source name")
header["srcname"] = ("N/A", "source name")
header["coh_dm"] = (dm, "coherent dispersion measure")
header["fchan1"] = (0, "frequency of channel 1")
header["foff"] = (0, "frequency width of a channel")
header["fchanbase"] = ("mid", "base point on channel (top, bottom, mid)")
header["badchan"] = ("", "comma-separated list of bad channels")
header["telescop"] = ("N/A", "telescope name")
header["ra"] = (0, "source right ascension")
header["dec"] = (0, "source declination")
header["tstart"] = (0, "time zero in MJD")
header["tsamp"] = (64e-6, "sampling time interval in seconds")
header["observer"] = ("amsterdm", "observer")
header["polchan"] = ("xx-yy", "polarization channel type")
hdu = pyfits.PrimaryHDU(header=header, data=simdata)
hdu.writeto("amsterdm-sim.fits")


print(simdata.shape)
print(times)
print(freqs)
print(simdata[..., 0, 0].mean())
print(simdata[..., 0, 1].mean())
print(simdata[..., 0, 2].mean())
plt.figure()
print(
    "stats before:", np.percentile(simdata, [0, 50, 100]), simdata.mean(), simdata.max()
)
plt.imshow(simdata[:, 0:2, ...].sum(axis=1).T, extent=[0, 1, 0, 1], vmin=0, vmax=160)
plt.colorbar()
plt.savefig("simdata-sim-before.png")
plt.clf()

# Calculate and correct bandpass
n = simdata.shape[0]
bkg_section = times > 0.4
print(n, bkg_section.sum())
print(simdata[bkg_section, ...].shape)
print(simdata[bkg_section, 0, 0].mean())
print(simdata[bkg_section, 0, 1].mean())
print(simdata[bkg_section, 0, 2].mean())
bandpass = simdata[bkg_section, ...].mean(axis=0)
print(f"{bandpass.shape = }")
print(bandpass[0])
dynspec = simdata / bandpass[None, ...]

# Combine X and Y polarization channels
print(dynspec.shape, dynspec[:, 0:2, :].shape)
dynspec = dynspec[:, 0:2, :].sum(axis=1)
# dynspec = dynspec[:, 0, :]**2 + dynspec[:, 1, :]**2

plt.figure()
print(
    "stats after:", np.percentile(dynspec, [0, 50, 100]), dynspec.mean(), dynspec.max()
)
plt.imshow(dynspec.T, extent=[0, 1, 0, 1], vmin=-0.3, vmax=2.7)
plt.colorbar()
plt.savefig("dynspec-sim-bandpasscorr.png")
plt.clf()

# Combine channels into a light curve
lc = dynspec.sum(axis=1)
print(lc.shape)


plt.figure()
plt.plot(times, lc)
plt.savefig("lc-sim.png")
