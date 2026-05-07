from contextlib import suppress
from functools import cached_property
import logging
from io import BufferedIOBase
from pathlib import Path, PurePath
from types import EllipsisType
import warnings

from astropy import units
from astropy.units import Quantity
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import core
from . import plot
from .constants import DEFAULT_BACKGROUND_RANGE, DMUNIT
from .io import read_fileformat, read_filterbank, read_fits, read_psrfits, read_hdf5
from .utils import FInterval


# Type alias
type array = np.ndarray | np.ma.MaskedArray

logger = logging.getLogger(__package__)


class Burst:
    """An FRB object"""

    @classmethod
    def fromfile(cls, fobj: BufferedIOBase):
        if isinstance(fobj, (str, PurePath)):
            path = Path(fobj)
            if not path.exists():
                raise IOError(f"{fobj} does not exist")
            if not path.is_file():
                raise IOError(f"{fobj} is not a file")
            fobj = path.open(mode="rb")
        if not isinstance(fobj, BufferedIOBase):
            raise IOError("object is not a valid binary I/O object")
        fileformat = read_fileformat(fobj)
        if fileformat == "filterbank-le":
            header, data = read_filterbank(fobj)
        elif fileformat == "filterbank-be":
            header, data = read_filterbank(fobj, le=False)
        elif fileformat == "fits":
            header, data = read_fits(fobj)
        elif fileformat == "psrfits":
            header, data = read_psrfits(fobj)
        elif fileformat == "hdf5":
            header, data = read_hdf5(fobj)
        header["format"] = fileformat

        return cls(header, data, file=fobj)

    def __init__(self, header, data, dm=0, file=None, copy=False):
        self.header = header.copy()
        self.data = data.copy() if copy else data
        self.dm = dm
        if not isinstance(self.dm, Quantity):
            self.dm *= DMUNIT
        # copy=False only works if `self.data` is already a MaskedArray
        self.data = np.ma.masked_invalid(self.data, copy=False)
        self._data = None
        self._dedispersed = None
        self._file = file
        if file:
            self.path = Path(self._file.name)
            self.filename = self.path.name
        else:
            self.path = self.filename = None
        self.badchannels = []

        self._fix_missing()
        self._set_attrs()
        self._flag_channels()

    # Make the class a context manager to support the 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _fix_missing(self):
        """Try and fix any missing keywords"""
        if "nchans" not in self.header:
            warnings.warn("'nchans' not found in header; determining from the data")
            self.header["nchans"] = self.data.shape[-1]
        if "fanchor" not in self.header:
            self.header["fanchor"] = "mid"
        if "fch1" not in self.header:
            if "fchan1" in self.header:
                self.header["fch1"] = self.header["fchan1"]
            elif "freqs" in self.header and isinstance(
                self.header["freqs"], (list, np.ndarray)
            ):
                logger.info("Determining 'fch1' key from frequency list")
                self.header["fch1"] = self.header["freqs"][0]
            else:
                logger.critical(
                    "'fch1' or related keyword not found in header; data can't be used"
                )
                raise ValueError(
                    "'fch1' or related keyword not found in header information"
                )
        if "foff" not in self.header:
            if "chan_bw" in self.header:
                self.header["foff"] = self.header["chan_bw"]
            else:
                logger.critical(
                    "'foff' or related keyword not found in header information"
                )
                raise ValueError(
                    "'foff' or related keyword not found in header information"
                )
        if "tsamp" not in self.header:
            if "tbin" in self.header:
                self.header["tsamp"] = self.header["tbin"]
            else:
                logger.critical(
                    "'tsamp' or related keyword not found in header information"
                )
                raise ValueError(
                    "'tsamp' or related keyword not found in header information"
                )

    def _set_attrs(self):
        """Set essential attributes from the header information"""

        # Time units in milliseconds
        # Frequency units in MegaHertz

        # sample time; time resolution.
        # Normally, we'd assume milliseconds, but headers tend
        # to be in seconds
        self._tsamp = self.header["tsamp"] * units.second
        # start time of observations
        self._tstart = self.header.get("tstart", 0) * units.day
        # frequency starting point, corresponding to channel 0
        self._fch1 = self.header["fch1"] * units.MHz
        # frequency interval; frequency resolution
        self._foff = self.header["foff"] * units.MHz
        # frequency anchor point: mid, bottom, top of channel
        # top means the higher frequency, bottom is the lower frequency of the channel
        self._fanchor = self.header["fanchor"]
        # number of channels in the bad
        self.nchans = self.header["nchans"]

    def _flag_channels(self):
        """Mask any invalid data, and set `self.badchannels`
        if a complete channel is masked"""
        if self.data.mask.any():
            for i in range(self.data.shape[-1]):
                if self.data[..., i].mask.all():
                    self.badchannels.append(i)

    # The following are properties, so that changing them will also change the
    # `times`, `reltimes`, `freqs`, `freq_offset` and `cfreq` (cached) properties
    @property
    def tsamp(self):
        return self._tsamp

    @tsamp.setter
    def tsamp(self, value):
        self._tsamp = value
        if self._tsamp is not None and not isinstance(self._tsamp, Quantity):
            self._tsamp = value * units.millisecond
        # Ensure the times property will be recalculated
        with suppress(AttributeError):
            del self.times
            del self.reltimes

    @property
    def tstart(self):
        return self._tstart

    @tstart.setter
    def tstart(self, value):
        self._tstart = value
        if self._tstart is not None and not isinstance(self._tstart, Quantity):
            self._tstart = value * units.day
        # Ensure the times property will be recalculated
        with suppress(AttributeError):
            del self.times
            del self.reltimes

    @property
    def fch1(self):
        return self._fch1

    @fch1.setter
    def fch1(self, value):
        self._fch1 = value
        if self._fch1 is not None and not isinstance(self._fch1, Quantity):
            self._fch1 *= units.MHz
        with suppress(AttributeError):
            del self.freqs
            del self.freq_offset

    @property
    def foff(self):
        return self._foff

    @foff.setter
    def foff(self, value):
        self._foff = value
        if self._foff is not None and not isinstance(self._foff, Quantity):
            self._foff *= units.MHz
        with suppress(AttributeError):
            del self.freqs
            del self.freq_offset

    @property
    def fanchor(self):
        return self._fanchor

    @fanchor.setter
    def fanchor(self, value):
        self._fanchor = value
        with suppress(AttributeError):
            del self.freqs
            del self.freq_offset

    # Cached properties only get calculated once.
    # To recalculate them, delete the attribute
    @cached_property
    def freq_offset(self):
        """Get the central point offset in the first channel"""
        offset = 0
        direc = 0
        if self.fanchor == "top":  # anchor at the higher frequency side
            direc = 1
        elif self.fanchor == "bottom":  # anchor at the lower frequency side
            direc = -1
        if self.foff < 0:
            offset = -direc * self.foff / 2
        else:
            offset = direc * self.foff / 2
        return offset

    @property
    def cfreq(self):
        """Central frequency"""
        midchan = self.nchans / 2
        cfreq = self.fch1 + self.freq_offset + midchan * self.foff
        return cfreq

    @cached_property
    def freqs(self):
        if "freqs" in self.header:
            return self.header["freqs"]
        freqs = self.fch1 + self.freq_offset + np.arange(self.nchans) * self.foff
        return freqs

    @cached_property
    def times(self):
        """Times in MJD

        Use the `reltimes` property for higher resolution timestamps
        """

        start = self.tstart
        dt = self.tsamp.to(units.day)
        nsamp = self.data.shape[0]
        times = start + np.arange(nsamp) * dt
        return times

    @cached_property
    def reltimes(self):
        """Relative times in seconds since start of observation"""

        if "reltimes" in self.header:
            return self.header["reltimes"]
        nsamp = self.data.shape[0]
        times = np.arange(nsamp) * self.tsamp
        return times

    def channel2freq(self, channel):
        freq = self.fch1 + self.freq_offset + channel * self.foff
        return freq

    def freq2channel(self, freq):
        channel = np.round((freq - self.fch1 - self.freq_offset) / self.foff)
        return channel

    def sample2time(self, sample):
        start = self.tstart
        dt = self.tsamp.to(units.day)
        time = start + sample * dt
        return time

    def time2sample(self, time):
        start = self.tstart
        dt = self.tsamp.to(units.day)
        sample = np.round((time - start) / dt)
        return sample

    def close(self):
        """Close the underlying file object"""
        if self._file and hasattr(self._file, "close"):
            self._file.close()

    def trim(
        self,
        times: tuple[float, float] | None = None,
        freqs: tuple[float, float] | None = None,
    ):
        """Trim the burst section to `times` and `freqs`

        Data is modified in-place. The data in the file, if it exists,
        is not touched.

        This action is non-reversible, except by recreating the Burst
        instance from the original data file.

        Parameters
        ----------

        times: 2-element tuple or list of floats, with start and end
            times in milliseconds. If `None`, no trimming is applied
            to the time axis.

        freqs: 2-element tuple or list of floats, with start and end
            frequencies in MegaHertz. If `None`, no trimming is
            applied to the frequency axis.

        """

        if times:
            dt = self.tsamp / 1000
            section = round(times[0] / 1e3 / dt), round(times[1] / 1e3 / dt)
            section = max(section[0], 0), min(section[1], self.data.shape[0])
            section = slice(*section)
            self.data = self.data[section, ...]
            if "reltimes" in self.header:
                self.header["reltimes"] = self.header["reltimes"][section]

        if freqs:
            section = (
                round(self.freq2channel(freqs[0])),
                round(self.freq2channel(freqs[1])),
            )
            if section[0] > section[1]:
                section = section[::-1]
            section = max(section[0], 0), min(section[1], self.data.shape[-1])
            section = slice(*section)
            self.data = self.data[..., section]
            if "freqs" in self.header:
                self.header["freqs"] = self.header["freqs"][section]

    def downsample(
        self, factor: int = 1, remainder: str = "droptail", method: str = "mean"
    ):
        """Downsample the data by `factor` along the sample/time dimension. Bins
        can be averaged (default) or summed together. The
        corresponding sample interval and times property are resampled
        accordingly.

        If the sample/time dimension doesn't match an integer number
        of `factor`, the remainder can be dropped, either from the
        start ("drophead") or the end ("droptail"; the default); or
        the remainder can be added to the last bin ("addtail") or be
        added to the first bin ("addhead").

        If the number of available bins in the data is smaller than
        `factor`, all bins are combined, even when `method` is one of
        "droptail" or "drophead".

        Raises a `ValueError`
            - for an incorrect `factor` (less than 1)
            - for an incorrect remainder value
            - for an incorrect method

        """

        self.data = core.downsample(
            self.data, factor=factor, remainder=remainder, method=method
        )
        self.tsamp *= factor
        # Clear the times and reltimes cached properties by deleting
        # it (if it was never used before, it won't exist: ignore that case).
        with suppress(AttributeError):
            del self.times
        with suppress(AttributeError):
            del self.reltimes

    def upsample(self, factor: int = 1):
        """Rebin the data to a higher resolution along the sample/time
        dimension. The sampling interval and times property are
        adjusted accordingly.

        Sample bins are simply split into `factor` new bins, with the same
        value as that of the original bin.

        Under the hood, this simply uses `numpy.repeat` for the first
        axis.

        """

        self.data = core.upsample(self.data, factor=factor)
        self.tsamp /= factor
        # Clear the times and reltimes cached properties by deleting
        # it (if it was never used before, it won't exist: ignore that case).
        with suppress(AttributeError):
            del self.times
        with suppress(AttributeError):
            del self.reltimes

    def flag(self, badchannels: set | list | np.ndarray | None = None):
        """Flag bad channels

        All given channels, corresponding to the array indices of the
        frequency axis, are masked along the time samples. This is
        done by using a NumPy masked array.

        The data is modified internally. The mask, however, is easily
        reverted or turned off completely

        This operation may turn the data into a MaskedArray, if it
        wasn't already a MaskedArray.

        If `badchannels` is empty or None, no operation is performed.

        """

        if badchannels:
            self.data = core.flag(self.data, badchannels)

    def dedisperse(self, dm: float | None):
        """Dedisperse the data for a given `dm`

        The data is internally dedispersed

        To prevent data from being dedispersed multiple data, a
        private copy of the original data is kept, and a flag is
        set. Any future dedispersion will be applied to the copy

        """

        if dm:
            if self._data is not None:
                data = self._data.copy()
            else:
                self._data = self.data.copy()
                data = self.data
            self.data = core.dedisperse(data, self.freqs, self.tsamp, dm)
            self._dedispersed = dm

    def calc_background(
        self,
        dm: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        method: str = "mean",
    ) -> tuple[array, array]:
        """Return background and its standard deviation for each channel

        This will flag the data if `badchannels` is given

        This will also dedisperses the data (which is intrinsic, so
        the data will be altered), unless `dm` is 0 or None.

        For details, see `core.calc_background`

        Returns
        -------
        Tuple of 2 arrays
            The background value and standard deviation across all frequency channels

        """

        self.flag(badchannels)
        if dm:
            self.dedisperse(dm)

        return core.calc_background(
            self.data, backgroundrange=backgroundrange, method=method
        )

    def dynspectrum(
        self,
        dm: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        background: tuple[float | dict, float | dict] | None = None,
    ):
        """Returns a dynamical spectrum for a given dispersion measure
        with the Stokes I / intensity parameter from the input data
        data array

        The routine flags bad channels, corrects for the given dispersion,
        calculates a background and corrects for the bandpass.

        When multiple polarization channels (xx and yy) exist, it does
        this for each independently, then combines the values together.

        It will optionally correct for bad channels, bandpass and dispersion, if
        the relevant keyword argument is given.

        Parameters
        ----------

        dm : float, optional

            Disperson measure

            Dedisperse the data for the given value.

            If set to None, the default value, the internal `dm`
            attribute of the burst is used. If this is also None, a
            warning is issued and no dedispersion is applied.

        badchannels : set | list | np.ndarray | None, default=None
            means no flagging is done.

            The bad channels are assumed to be the same for the xx and yy
            polarizations, if applicable.

        backgroundrange: 2-tuple, or iterable of 2-tuples

            Interval, or iterable of intervals, as fractions of the
            sample dimension of the data, that is, each interval item
            contains a start and end fraction of the first dimension
            of the data that corresponds to a background area

        bkg_extra : bool, default=False

            If ``True``, returns an additional object, which is a dict containing
            the mean and standard deviation of the background along the channels;
            these are one-dimensional arrays

        background: tuple of mean and standard deviation of the background values

            The tuple values can also be dicts. In that case, the keys are
            the polarization keys, (xx and yy), with the valuse the mean
            and standard deviation for those polarization parts. If the
            tuple elements are single values, but the input data contains
            multiple polarizations, it is assumed that the mean and
            standard deviation are the same for xx and yy.


        If the `background` argument is given, `backgroundrange` and
        `bkg_method` are ignored.

        Returns
        -------
            The dynamical spectrum: a two-dimensional array with the
            Stokes intensity parameter

        """

        if self.header.get("pol_type", "").lower() == "iquv":
            # Four polarization channels; use only stokes I
            data = self.data[:, 2, :]
            logger.info("Selecting Stokes I data for dynamic spectrum")
        else:
            data = self.data

        data = core.flag(data, badchannels)

        if dm is None:
            dm = self.dm
            if dm is None:
                warnings.warn("No `dm` supplied and no default dm available")

        dynspec, _ = core.create_dynspectrum(
            data,
            self.freqs,
            self.tsamp,
            dm,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
        )

        return dynspec

    def calc_intensity(
        self,
        dm: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        datarange: tuple[float, float] | None = None,
        bkg_extra: bool = False,
    ):
        """Returns the Stokes I parameter from the xx and yy signals

        It will optionally correct for bad channels, bandpass and
        dispersion, if the relevant keyword argument is given.

        .. deprecated::
            use `create_dynspectrum` instead.

        Parameters
        ----------

        badchannels : set, list or array of channel indices to flag. The default of None
            means no flagging is done.

        datarange : two-tuple of floating point fractions between 0 and 1

            Fractional range along the time axis, where the actual object
            is located. Data outside these columns is used for the
            bandpass correction.

            The default of None indicates no bandpass correction is applied.

        dm : float, optional

            Disperson measure

            Dedisperse the data for the given value.

            If set to None, the default value, the internal `dm`
            attribute of the burst is used. If this is also None, a
            warning is issued and no dedispersion is applied.

        bkg_extra: bool, default False

            If `True`, returns an additional object, which is a dict
            containing the mean and standard deviation of the background
            along the channels; these are one-dimensional arrays


        Returns
        -------

        Two-dimensional array with the Stokes intensity parameter. If
        `bkg_extra` is `True`, returns a two-tuple of (two-dimensional
        array, bkg_info dict).

        """

        data = dict(xx=self.data[:, 0, :], yy=self.data[:, 1, :])

        if dm is None:
            dm = self.dm
            if dm is None:
                warnings.warn("No `dm` supplied and no default dm available")

        if dm:
            dm = {"dm": dm, "freq": self.freqs, "tsamp": self.tsamp}

        intensity = core.calc_intensity(
            data, dm, badchannels, datarange, bkg_extra=bkg_extra
        )

        return intensity

    def lightcurve(
        self,
        dm: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
    ):
        data = core.flag(self.data, badchannels)

        if dm is None:
            dm = self.dm
            if dm is None:
                warnings.warn("No `dm` supplied and no default dm available")

        lightcurve, _ = core.calc_lightcurve(
            data,
            self.freqs,
            self.tsamp,
            dm,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
        )

        return lightcurve

    def bowtie(
        self,
        dminterval: FInterval,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        ndm: int = 50,
        reffreq: float | None = None,
    ) -> np.ndarray:
        data = core.flag(self.data, badchannels)

        return core.bowtie(
            data,
            self.freqs,
            self.tsamp,
            dminterval,
            reffreq=reffreq,
            ndm=ndm,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
        )

    def signal2noise(
        self,
        dminterval: FInterval,
        dm: float | None = None,
        reffreq: float | None = None,
        ndm: int = 50,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        background: tuple[float | dict, float | dict] = None,
        peak: bool = True,
    ) -> tuple[array, array]:
        data = core.flag(self.data, badchannels)

        dms, ratios = core.signal2noise(
            data,
            self.freqs,
            self.tsamp,
            dminterval,
            dm=dm,
            reffreq=reffreq,
            ndm=ndm,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
            background=background,
            peak=peak,
        )

        return dms, ratios

    def waterfall(
        self,
        dm: float | Quantity | None = None,
        reffreq: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        ax: Axes | None = None,
        **options,
    ) -> tuple[Figure, Axes]:
        """Return a waterfall plot for the burst

        Parameters
        ----------

        See `plot.waterfall` for a description of most parameters.

        dm : float, optional

            Disperson measure

            Dedisperse the data for the given value.

            If set to None, the default value, the internal `dm`
            attribute of the burst is used. If this is also None, a
            warning is issued and no dedispersion is applied.


        Returns
        -------
        A tuple of [Figure, Axes]

        """

        if dm is None:
            dm = self.dm
            if dm is None:
                warnings.warn("No `dm` supplied and no default dm available")
                dm = 0 * DMUNIT

        if self.data.ndim == 3:
            if self.data.shape[1] != 2:  # Assume Stokes IQUV
                data = self.data[:, 0, :]
            # else assume xx and yy
        else:
            data = self.data

        return plot.waterfall(
            data,
            self.freqs,
            self.tsamp,
            dm=dm,
            reffreq=reffreq,
            badchannels=badchannels,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
            ax=ax,
            **options,
        )

    def lcplot(
        self,
        dm: float | None = None,
        reffreq: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        ax: Axes | None = None,
        **options,
    ) -> tuple[Figure, Axes]:
        """Return a light curve plot for the burst

        Parameters
        ----------

        See `plot.lightcurve` for a description of most parameters.

        dm : float, optional

            Disperson measure

            Dedisperse the data for the given value.

            If set to None, the default value, the internal `dm`
            attribute of the burst is used. If this is also None, a
            warning is issued and no dedispersion is applied.


        Returns
        -------
        A tuple of [Figure, Axes]

        """

        if dm is None:
            dm = self.dm
            if dm is None:
                warnings.warn("No `dm` supplied and no default dm available")
                dm = 0

        if self.data.ndim == 3:
            if self.data.shape[1] != 2:  # Assume Stokes IQUV
                data = self.data[:, 0, :]
            # else assume xx and yy
        else:
            data = self.data

        return plot.lightcurve(
            data,
            self.freqs,
            self.tsamp,
            dm=dm,
            reffreq=reffreq,
            badchannels=badchannels,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
            ax=ax,
            **options,
        )

    def bgplot(
        self,
        dm: float | None = None,
        reffreq: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        ax: Axes | None = None,
        **options,
    ) -> tuple[Figure, Axes]:
        """Return a background plot for the burst

        This will plot the background, averaged over backgroundrange
        in the sample dimension. It will plot both the background and
        its standard deviation.

        Parameters
        ----------

        See `plot.background` for a description of most parameters.

        dm : float, optional

            Disperson measure

            Dedisperse the data for the given value.

            If set to None, the default value, the internal `dm`
            attribute of the burst is used. If this is also None, a
            warning is issued and no dedispersion is applied.


        Returns
        -------
        A tuple of [Figure, Axes]

        """

        if dm is None:
            dm = self.dm
            if dm is None:
                warnings.warn("No `dm` supplied and no default dm available")
                dm = 0

        return plot.background(
            self.data,
            self.freqs,
            self.tsamp,
            dm=dm,
            reffreq=reffreq,
            badchannels=badchannels,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
            ax=ax,
            **options,
        )

    def bowtieplot(
        self,
        dminterval: FInterval,
        reffreq: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        ndm: int = 50,
        trange: slice | EllipsisType = Ellipsis,
        ax: Axes | None = None,
        **options,
    ) -> tuple[Figure, Axes]:
        """Return a bowtie plot for the burst

        Parameters
        ----------

        See `plot.bowtie` for a description of most parameters.

        Returns
        -------
        A tuple of [Figure, Axes]

        """

        if self.data.ndim == 3:
            if self.data.shape[1] != 2:  # Assume Stokes IQUV
                data = self.data[:, 0, :]
            # else assume xx and yy
        else:
            data = self.data

        return plot.bowtie(
            data,
            self.freqs,
            self.tsamp,
            dminterval=dminterval,
            reffreq=reffreq,
            badchannels=badchannels,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
            ndm=ndm,
            trange=trange,
            ax=ax,
            **options,
        )

    def s2nplot(
        self,
        dminterval: FInterval,
        reffreq: float | None = None,
        ndm: int = 50,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        peak: bool = True,
        fit: bool = False,
        ax: Axes | None = None,
        **options,
    ) -> tuple[Figure, Axes]:
        """Return a signal-to-noise plot for the burst

        Parameters
        ----------

        See `plot.signal2noise` for a description of most parameters.

        Returns
        -------
        A tuple of [Figure, Axes]

        """

        if self.data.ndim == 3:
            if self.data.shape[1] != 2:  # Assume Stokes IQUV
                data = self.data[:, 0, :]
            # else assume xx and yy
        else:
            data = self.data

        return plot.signal2noise(
            data,
            self.freqs,
            self.tsamp,
            dminterval=dminterval,
            reffreq=reffreq,
            ndm=ndm,
            badchannels=badchannels,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
            peak=peak,
            fit=fit,
            ax=ax,
            **options,
        )

    def dmplot(
        self,
        dminterval: FInterval,
        dm: float | None = None,
        reffreq: float | None = None,
        ndm: int = 50,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "mean",
        peak: bool = True,
        dm_coherent: float = 0,
        ax: Axes | None = None,
        **options,
    ) -> tuple[Figure, Axes]:
        """Plot a combination of a waterfall plot, a light curve and background

        Parameters
        ----------

        See `plot.grid` for a description of most parameters.

        dm : float, optional

            Disperson measure

            Plot the data at the given dedispersion.

            If set to None, the default value, the internal `dm`
            attribute of the burst is used. If this is also None, a
            warning is issued and no dedispersion is applied.

            `dminterval` is used for the signal-to-noise plot, while
            `dm_coherent` is used to calculate the smearing.

        Returns
        -------
        A tuple of [Figure, Axes]

        """

        if dm is None:
            dm = self.dm
            if dm is None:
                warnings.warn("No `dm` supplied and no default dm available")
                dm = 0

        data = self.data[:, 0, :] if self.data.ndim == 3 else self.data

        return plot.grid(
            data,
            self.freqs,
            self.tsamp,
            dm=dm,
            dminterval=dminterval,
            reffreq=reffreq,
            ndm=ndm,
            badchannels=badchannels,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
            peak=peak,
            dm_coherent=dm_coherent,
            foff=self.foff,
            cfreq=self.cfreq,
            ax=ax,
            **options,
        )


def open(name: Path | str):
    """Helper method that returns an open file

    Can be use with a context manager:

        with open(path) as burst:
            ...

    """

    return Burst.fromfile(name)
