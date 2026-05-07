# The base modules

The core and burst modules form the basis of the AmsterDM package. The core module contains a set of practical functions to dedisperse data, calculate a dynamic spectrum, a light curve or a bow-tie plot, while the burst module contains a Burst class that provides a more object-oriented interface to the data, packing all necessary details into a single Burst instance.

## The core module

The core module contains a set of functions that act on data as two- or three-dimensional NumPy arrays representing the actual data. The frequencies and sampling interval are also input, as are the dispersion measure and an optional dedispersion reference frequency. Some functions require a dispersion measure interval, such as those for creating the data for a bow-tie plot or for estimating and fitting signal-to-noise data. Details for each function are in the API documentation.

The data NumPy array uses (time) samples for the first axis and channel (frequency) for the last axis. For a two-dimensional array, these are the only axes; for a three-dimensional array, the center axis is for the polarization channels. The latter can be of size 1, in which case it is assumed to be Stokes I and is equivalent to a two-dimensional array; size 2, which assumes xx and yy polarization channels; and size 4, which assumes Stokes I, Q, U and V polarization channels. For xx-yy polarization, some functions will correct (dedisperse and bandpass-normalize) the channels individually, then add them together, to produce, for example, a dynamic spectrum or a light curve. For a set of four-Stokes channels, only the first channel is used and Q, U and V are simply ignored.

The frequency input is a one-dimensional list or array that matches the frequency axis of the input data. The sampling interval is a simple float, while the optional reference frequency is also a float. The dispersion measure is a float, while any interval is a tuple or list of two floats, start and end.

All values, except for the data array itself, can (and preferably should have) units attached to them. This can be done with `astropy.units`, which is used under the hood. This ensures correct interpretation of input values, so that no mistakes between, e.g., seconds and milliseconds are made. If no units are given, frequencies are assumed to be in megahertz (MHz), sampling time in milliseconds (ms) and dispersion measure in cm^-3 pc. For the latter, a constant exists that is this exact unit, `amsterdm.constants.DMUNIT`.


## The burst module

The burst module wraps the above functionality into a `Burst` class that keeps track of the data, frequencies, sampling time and other details. It is usually created from an input data file, either with the `Burst.fromfile` class method or with the basic `open()` function. The header of the file should provide enough input to automatically set the frequencies and sampling time correctly, but sometimes manual adjustment may be needed.

When using the `open()` function for reading a data file, this can be used as a context manager with the `with` statement, e.g. 

```python
with open("somefile.dat") as burst:
    ...  # execute code with burst
```

Sample time and frequency information for a `Burst` instance is not set directly with a frequency array, but through specific attributes (corresponding to somewhat standard header keywords). These are

- `tsamp` for the sample time interval. If no unit is given, milliseconds are assumed.
- `tstart` for the starting date-time point of the first sample. If not unit is given, days are assumed.
- `fch1` for the frequency of the first channel. If no unit is given, MHz is assumed.
- `foff` for the offset between individual channels. This can be negative. If no unit is given, MHz is assumed.
- `fanchor` for the anchoring point of the first frequency `fch1`. That is, `"mid"` for the center of the channel, `"bottom"` for the lower-frequency side of the channel (even with a negative `foff`) and `"top"` for the upper-frequency side of the channel.

The above properties can be read and set. The following properties are intended to be read-only:

- `times` is an array of timestamps of the samples, using `tsamp` and `tstart`.
- `reltimes` is an array of timestamps of the samples, assuming 0 for the first sample, in seconds.
- `freqs` is an array of frequencies matching the channels, using `fch1`, `foff` and `fanchor`.
- `cfreq` is the central frequency, essentially the midpoint of `freqs`.

### Burst core and plotting methods

A `Burst` instance has methods that match the core functions, without the need to supply the data, frequencies and sampling interval manually, as this is part of the `Burst` instance.

Similarly, there are various plotting routines available as methods. These routines don't directly create the figure; that is, they don't save a figure to file or open a new window with the plot. Instead, these routines return a tuple of a Matplotlib `Figure` instance and an `Axes` instance. This is often enough in a notebook to immediately plot the figure, but in a script, the figure may still need to be saved manually. For example

```python
with open("myburst.dat") as burst:
    fig, ax = burst.waterfall()
    fig.savefig("myburst.png")
```

would do that.

The plotting methods also take an optional `ax` keyword argument, that contains a Matplotlib `Axes` instance to draw on. Otherwise, the figure and axes are created on the fly inside the method.

Existing plotting methods are 

- `waterfall` creates a waterfall plot (dynamic spectrum plot)
- `lcplot` creates a light curve
- `bgplot` creates a plot with the background summed across the samples, as well as its standard deviation (so background versus frequency)
- `bowtieplot` creates a bow-tie plot
- `s2nplot` creates a signal-to-noise plot, optionally with a Gaussian fit
- `dmplot` creates a combination of a waterfall plot, a light curve and a background plot

The various plotting methods also take a set of keyword arguments for tweaking some aspects of the plots themselves, similar to Matplotlib's functions. Examples are a color-normalization range (`vmin` and `vmax`), logarithmic axes (`logscale`) or x- and y-labels (`xlabel`, `ylabel`). See the `examples/plots.py` example, or the `amsterdm.plot` module for more details.
