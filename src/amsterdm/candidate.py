from contextlib import suppress
from functools import cached_property
from io import BufferedIOBase
from pathlib import Path, PurePath

import numpy as np

from . import core
from .constants import DEFAULT_BACKGROUND_RANGE, SOD
from .io import read_fileformat, read_filterbank, read_fits
from .utils import FInterval


class Candidate:
    """A candidate FRB object"""

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
        header["format"] = fileformat

        return cls(header, data, fobj)

    def __init__(self, header, data, file=None, copy=False):
        self.header = header.copy()
        self.data = data.copy() if copy else data
        self._file = file
        self.path = Path(self._file.name)
        self.filename = self.path.name

        if "fanchor" not in self.header:
            self.header["fanchor"] = "mid"

    # Make the class a context manager to support the 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @cached_property
    def freqs(self):
        nfreq = self.header["nchans"]
        foff = self.header["foff"]
        start = self.header["fch1"]
        fanchor = self.header["fanchor"]
        # Calculate the central point offset in the first channel
        offset = 0
        direc = 0
        if fanchor == "top":  # anchor at the higher frequency side
            direc = 1
        elif fanchor == "bottom":  # anchor at the lower frequency side
            direc = -1
        if foff < 0:
            offset = direc * foff / 2
        else:
            offset = -direc * foff / 2
        freqs = start + offset + np.arange(nfreq) * foff
        return freqs

    @cached_property
    def times(self):
        """Times in MJD

        Use the `reltimes` property for higher resolution timestamps
        """

        start = self.header["tstart"]
        dt = self.header["tsamp"] / SOD
        nsamp = self.data.shape[0]
        times = start + np.arange(nsamp) * dt
        return times

    @cached_property
    def reltimes(self):
        """Relative times in seconds"""
        dt = self.header["tsamp"]
        nsamp = self.data.shape[0]
        times = np.arange(nsamp) * dt
        return times

    def channel2freq(self, channel):
        foff = self.header["foff"]
        start = self.header["fch1"]
        freq = start + channel * foff
        return freq

    def freq2channel(self, freq):
        foff = self.header["foff"]
        start = self.header["fch1"]
        channel = np.round((freq - start) / foff)
        return channel

    def sample2time(self, sample):
        start = self.header["tstart"]
        dt = self.header["tsamp"] / SOD
        time = start + sample * dt
        return time

    def time2sample(self, time):
        start = self.header["tstart"]
        dt = self.header["tsamp"] / SOD
        sample = np.round((time - start) / dt)
        return sample

    def close(self):
        """Close the underlying file object"""
        if self._file and hasattr(self._file, "close"):
            self._file.close()

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
        self.header["tsamp"] *= factor
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
        self.header["tsamp"] /= factor
        # Clear the times and reltimes cached properties by deleting
        # it (if it was never used before, it won't exist: ignore that case).
        with suppress(AttributeError):
            del self.times
        with suppress(AttributeError):
            del self.reltimes

    def create_dynspectrum(
        self,
        dm: float,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "median",
        bkg_extra: bool = False,
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

        dm : float

            Disperson measure

            Dedisperse the data for the given value. The default value of
            None means no dedispersion is applied.


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
        `bkg_method` are ignored. If `bkg_extra` is also set, the returned
        values identical to the given values.

        Returns
        -------

            Two-dimensional array with the Stokes intensity parameter. If
            ``bkg_extra`` is ``True``, returns a two-tuple of (two-dimensional
            array, bkg_info dict).

        """

        if dm:
            dm = {"dm": dm, "freq": self.freqs, "tsamp": self.header["tsamp"]}

        dynspec = core.create_dynspectrum(
            self.data,
            dm,
            badchannels,
            backgroundrange,
            bkg_method=bkg_method,
            bkg_extra=bkg_extra,
        )

        return dynspec

    def calc_intensity(
        self,
        dm: float,
        badchan: set | list | np.ndarray | None = None,
        datarange: tuple[float, float] | None = None,
        bkg_extra: bool = False,
    ):
        """Returns the Stokes I parameter from the xx and yy signals

        It will optionally correct for bad channels, bandpass and
        dispersion, if the relevant keyword argument is given.

        Parameters
        ----------

        badchan : set, list or array of channel indices to flag. The default of None
            means no flagging is done.

        datarange : two-tuple of floating point fractions between 0 and 1

            Fractional range along the time axis, where the actual object
            is located. Data outside these columns is used for the
            bandpass correction.

            The default of None indicates no bandpass correction is applied.

        dm : float

            Disperson measure

            Dedisperse the data for the given value. The default value of
            None means no dedispersion is applied.

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

        if dm:
            dm = {"dm": dm, "freq": self.freqs, "tsamp": self.header["tsamp"]}

        intensity = core.calc_intensity(
            data, dm, badchan, datarange, bkg_extra=bkg_extra
        )

        return intensity

    def lightcurve(
        self,
        dm: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "median",
        bkg_extra: bool = False,
    ):
        if dm:
            dm = {"dm": dm, "freq": self.freqs, "tsamp": self.header["tsamp"]}

        lightcurve = core.calc_lightcurve(
            self.data,
            dm,
            badchannels,
            backgroundrange,
            bkg_method=bkg_method,
            bkg_extra=bkg_extra,
        )

        return lightcurve

    def bowtie(
        self,
        dm: FInterval,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "median",
        ndm: int = 50,
    ) -> np.ndarray:
        return core.bowtie(
            self.data,
            dm,
            self.freqs,
            self.header["tsamp"],
            badchannels,
            backgroundrange,
            bkg_method=bkg_method,
            ndm=ndm,
        )

    def signal2noise(
        self,
        dms: np.ndarray,
        reffreq: float | None = None,
        badchannels: set | list | np.ndarray | None = None,
        backgroundrange: FInterval | tuple[FInterval] = DEFAULT_BACKGROUND_RANGE,
        bkg_method: str = "median",
        background: tuple[float | dict, float | dict] = None,
        peak: bool = True,
        peak_interval: FInterval | None = None,
    ):
        ratios = core.signal2noise(
            self.data,
            dms=dms,
            freqs=self.freqs,
            dtsamp=self.header["tsamp"],
            reffreq=reffreq,
            badchannels=badchannels,
            backgroundrange=backgroundrange,
            bkg_method=bkg_method,
            background=background,
            peak=peak,
            peak_interval=peak_interval,
        )

        return ratios


def openfile(name: Path | str):
    return Candidate.fromfile(name)
