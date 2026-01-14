from functools import cached_property
from io import BufferedIOBase
from pathlib import Path, PurePath
from types import EllipsisType

import numpy as np

from .core import calc_intensity
from .io import read_fileformat, read_filterbank, read_fits


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

    # Make the class a context manager to support the 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @cached_property
    def freqs(self):
        nfreq = self.header["nchans"]
        foff = self.header["foff"]
        start = self.header["fch1"] + foff / 2
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

    def calc_intensity(
        self,
        dm: dict[str, float | np.ndarray] | None = None,
        badchan: set | list | np.ndarray | None = None,
        datarange: tuple[float, float] | None = None,
        samplerange: slice | EllipsisType | None = None,
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

        if not samplerange:
            data = dict(xx=self.data[:, 0, :], yy=self.data[:, 1, :])
        else:
            data = dict(
                xx=self.data[samplerange, 0, :], yy=self.data[samplerange, 1, :]
            )

        if dm:
            dm = {"dm": dm, "freq": self.freqs, "tsamp": self.header["tsamp"]}
        intensity = calc_intensity(data, dm, badchan, datarange, bkg_extra=bkg_extra)

        return intensity


def openfile(name: Path | str):
    return Candidate.fromfile(name)
