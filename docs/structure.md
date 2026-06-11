# Notes on the structure optimization implementation

## Usage

For usage of the structure calculation, the code makes use of a `Structure` class. This allows it to easily store and access intermediate variables, that could then be plotted for verification plots.

The input data for the `Structure` class requires so-called bowtie data: light curve data at a range of dispersion measures, in the form of a two-dimensional array. Also input are the dispersion measures themselves, a one-dimensional array.

Sample code would then look as follows:

```
burst = Burst(header, data)  # or read directly from file with `burst = Burst.fromfile(path)`

tie, dms = burst.bowtie(dminterval=[-1, 1], backgroundrange=[0.5, 1], ndm=150)

struct = structure.Structure(tie, dms)

optdm, lowdm, highdm, mindm, maxdm = struct.calc()
```

`struct.calc()` will automatically calculate a best estimate for `kc`, the spectral cutoff in the low-pass filter. One could also set a value in the class constructor (keyword argument `kc`) or even as an attribute after instance creation. As long as `kc` is not `None` (or zero), a best estimate will be calculated. The order parameter is an optional argument to `struct.calc()`; its default is 3.

The `lowdm` and `highdm` values contain the errors on `optdm`, while `mindm` and `maxdm` contain the range (so `optdm` + `lowdm` equals `mindm` and `optdm` + `highdm` equals `maxdm`); note that `lowdm`, the lower error, is given as a negative value.

For plotting, after the calculation, one can do

```
fig, ax = struct.plot_spectrum()
fig.savefig("spectrum.png")

fig, ax = struct.plot_structure()
fig.savefig("structure.png")

fig, ax = struct.plot_adjusted_structure()
fig.savefig("adjusted_structure.png")

fig, ax = struct.plot_uncertainty()
fig.savefig("uncertainty.png")

fig, ax = struct.plot_detrended_noise()
fig.savefig("detrended_noise.png")

fig, ax = struct.plot_relative_detrended_noise()
fig.savefig("relative_detrended_noise.png")
```

For a more complete example, see `examples/structure.py`.

## Implementation notes

The code follows the article formulations, as well as the code given at <https://github.com/marcinglowacki/SHRINE>.

The section for estimating the range of DM (and the uncertainty) has some differences and unclarity:

- when the range is deduced in the original code, the first index *after* the adjusted structure parameter is *above* the structure parameter is included (which equates to the lower bound), while the first index *after* the adjusted structure parameter is *below* the structure parameter is also included (which equates to the upper bound). The means for the uncertainty indices, the first index is inside the interval, but the last index is just outside the interval. This shifts the interval slightly to the right (higher values). This is all under the assumption that the values are monotonically increasing, which is normally the case.

  In the AmsterDM, both sides are inside the interval. This shrinks the uncertainty interval slightly, but makes it symmetric. The results therefore differ at the upper bound.

  Perhaps more ideally, the interval edges should include half the bin size (DM bin width), or interpolate between the boundary indices inside and outside the interval.

  Both methods warn if there are gaps where the adjusted structure parameter temporarily is below the maximum structure parameter; in both cases, the final outer edges are used for the uncertainty interval. So the gaps are ignored in the end result of the uncertainty calculation.

- The actual uncertainty calculation is unclear. It redoes some steps as for the structure parameter calculation, but then for the relative noise. There is also a single discrete cosine transform, but no inverse; this means "ortho" is required for the normalization. The use of "ortho" is presumably because the eigenvectors should be normalised.

  There is also a double filter step in the uncertainty calculation: the bandpass filter is applied to data that already has had the lowpass filter applied. It is unclear why this is the case, and where in the article this shows up. Note that the SHRINE code mentions in the doc-string: `LPF_data: The discrete cosine space smoothed data.`, which would mean `C^T i`, but it actually is the smoothed data filtered by the low-pass filter (which the name `LPF_data` implies); and the calling code does indeed pass low-pass-filtered smoothed data as argument to this function.

  In AmsterDM, this functionality is used as well: low-pass filtered data is passed to the uncertainty calculation, and is there filtered again by the (full) bandpass.

  Probably, low-pass filtering first and then bandpass filtering doesn't change compared to only bandpass filtering; perhaps the effect on the lower side is stronger with double filtering, but data values in that regime should already be minimal and neglible after the first filter, sp that changes resulting from the second filter stay neglible.
