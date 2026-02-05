import astropy.io.fits as pyfits

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
dfreq = -2
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
header["fchan1"] = (1400, "frequency of channel 1 [MHz]")
header["foff"] = (dfreq, "frequency width of a channel")
header["fanchor"] = ("mid", "anchor point on channel (top, bottom, mid)")
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
