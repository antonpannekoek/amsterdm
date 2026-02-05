The use of AmsterDM in Python programs can partly be seen from the `examples/plots.py` example program. The utilities in AmsterDM try to be generic, in particular for plotting, but enhancements may be added over time.

The implementation of AmsterDM has a few building blocks:

- The `core` module contains core functionality (dedispersion, resampling, background estimation). The functions in this module form the bases of all other functionality, and be directly used. Nearly every function has as its first input a three-dimensional NumPy array, of size `<samples, nifs, channels>`, where `nifs` are the individual polarization channels (this can be one, i.e., Stokes I). Exceptions are functions that work on light curve data, which accept a one-dimensional array (generally, the channel-squashed dynamical spectrum).

  Core funcions are detailed in their doc-strings.

- The `burst` module wraps around an input data file. This does read the full file directly into memory, since most operations require the full data (for example, the background estimation)[^1]

  The wrapped data is contained in a `Burst` class instance; the methods are the same as (or very similar to) the core functions, just with the first argument skipped (as this is the instance).

  A `Burst` instance contains a `header` and a `data` attribute, which contain what their name suggest: header information in the form of a dict (or a FITS header), and the three-dimensional data. Note that by default, no header information is added to the class instance as attribute: it is felt better to clearly separate this information into a header, and let attributes be other pieces of information, such as properties described below.

  There are a few special properties: `freqs` and `times`, which are conversions from the channels and samples. Be aware that these are cached properties: if anything changes under the hood (e.g., you change the channel offset or the start time), the properties need to be recalculated: simply delete the property (e.g., `del burst.times`) and use it again: it will be recalculated automatically that way. Note that `times` are the MJD, deduced from the actual observation start time (`tstart` in the header).

- The `plot` module relies on a `Burst` instance as the first argument, and contains functions to create a variety of plots, using Matplotlib. The routines can also take an optional `ax` argument, on which the plot(s) will be drawn if given. If not, a new figure is created under the hood by the routines.

- The `sim` module contains a few standalone routines, for basic simulation of data; it doesn't rely on other functionality in the AmsterDM package. It is mostly used for simple testing of procedures.

- The `constants` modules contains a few practical constants. The `DMCONST` is probably the most important one. It can'be easily changed, but most functions that use this constant, accept another user-provided `dmconst` keyword argument (with `DMCONST` the default value).



[^1]: ignoring some polarization channels may be possible, but since this dimension is generally the central dimension (and therefore don't essentially speed up the reading), this is not a primary concern.
