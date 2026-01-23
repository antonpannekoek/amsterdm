from functools import cached_property
from io import BufferedIOBase
from pathlib import Path, PurePath

import numpy as np

from . import core
from .constants import DEFAULT_BACKGROUND_RANGE

from .io import read_fileformat, read_filterbank, read_fits
from .utils import FInterval


SOD = 60 * 60 * 24


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
        freqs = start + np.arange(nfreq) * foff
        return freqs

    @cached_property
    def times(self):
        start = self.header["tstart"]
        dt = self.header["tsamp"] / SOD
        nsamp = self.data.shape[0]
        times = start + np.arange(nsamp) * dt
        return times

    def close(self):
        """Close the underlying file object"""
        if self._file and hasattr(self._file, "close"):
            self._file.close()

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


def openfile(name: Path | str):
    return Candidate.fromfile(name)
