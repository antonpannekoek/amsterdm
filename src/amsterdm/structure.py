"""Module to compute a best estimate for the dispersion measure, including an error estimate.

Following the publication of Sutinjo et al, 2023,
10.3847/1538-4357/ace774 . Please cite this publication if you are
using this code.

Code adopted from initial code provided by T Perrott, with additions
from A Bera and M Glowacki and based on earlier code by by A Sutinjo
and a script by D Scott. See https://github.com/marcinglowacki/SHRINE

"""

import logging
import sys
import warnings

from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.fftpack import dct, idct

from .constants import Array
from .plot import ensure_figure, set_title_labels


logger = logging.getLogger(__package__)


def _warn_to_logger(message, category, filename, lineno, file=None, line=None):
    """Helper function to write warnings both by the logger, and to stderr"""

    logger.warning("%s:%d: %s: %s", filename, lineno, category.__name__, message)
    print(f"{category.__name__}: {message}", file=sys.stderr)


logger.showwarning = _warn_to_logger


class Structure:
    """Class to hold the structure data and calculation results

    While the actual calculation can be a straightforward function,
    additional functionality such as creating diagnostic plots require
    the intermediate calculated data. These are kept as part of the
    class, and then plots can be optionally created after the initial
    calculation.

    If you use this functionality of AmsterDM, please cite Sutinjo et
    al, 2023, 10.3847/1538-4357/ace774

    """

    def __init__(self, data: Array, dms: np.ndarray, kc: int | None = None):
        """Set up the structure data

        Parameters
        ----------
            data: two-dimensional array

                This should be "bowtie" data: a two-dimensional array of
                dispersion measures along the y-axis versus stacked time
                samples (i.e., light curves) along the y-axis.

            dms: one-dimensional array

                the dispersions measures corresponding to the y-axis
                of the data. This should be monotonically increasing.

            kc: float or None, default None

                Spectral cutoff value. If `None`, a best estimate will be
                computed using `calc_kc`; this is the default.

        """

        if data.ndim != 2:
            raise ValueError("input data is not two-dimensional")
        if data.shape[0] != len(dms):
            raise ValueError("`data` first/outer axis does not match `dms`")
        self.data = data.copy()
        if not all(np.diff(dms) > 0):
            raise ValueError("dms should be monotonically increasing")
        self.dms = dms.copy()
        self.kc = kc

        self.setup()

    def setup(self):
        """Calculation of some initial variables"""

        self.dctdata = dct(self.data, norm="ortho")
        ndm, self.nsamples = self.dctdata.shape
        if ndm != len(self.dms):
            raise ValueError(
                "Number of DMs in `dms` does not match the first (y) axis of the input data"
            )

    def calc(
        self,
        order: int = 3,
    ) -> tuple[Quantity, Quantity, Quantity, Quantity, Quantity]:
        """Optimize a DM by optimizing structure data; provide an error estimate on DM as well

        Parameters
        ----------

        order: positive integer, default 3

            Order of the low-pass filter function


        """

        self.kc = self.kc or self.calc_kc()  # dctdata, input_is_dct=True)
        logger.info("using kc = %d", self.kc)

        # Create the low-pass filter
        k = np.linspace(1, self.nsamples, self.nsamples)
        lowpass = 1 / (1 + (k / self.kc) ** (2 * order))  # Eq 17

        self.lowpass = np.diag(lowpass)  # diagonal matrix with the low-pass filter
        # smoothed dct data
        self.dctdata_lp = self.lowpass @ self.dctdata.T

        # Smoothed intensity ĩ (Si)
        self.smoothdata = idct(self.dctdata_lp.T, norm="ortho")  # Eq 16

        eigenvalues = 2 - 2 * np.cos((k - 1) * np.pi / self.nsamples)  # Eq 15
        highpass = np.sqrt(eigenvalues)

        bandpass = highpass * lowpass
        self.bandpass = np.diag(bandpass)
        bpdata = self.bandpass @ self.dctdata.T

        # Calculate the normed data
        # This is equivalent to the structure parameter
        self.structure = np.linalg.norm(bpdata, axis=0)

        # Maximum of the structure parameter is our best estimate for DM
        argmax = np.argmax(self.structure)
        self.maxstructure = self.structure[argmax]

        self.optdm = self.dms[argmax]

        # Calculate the uncertainty

        # detrended noise = noisy data - smoothed data
        self.noise = self.data - self.smoothdata
        self.deltanoise = self.noise - self.noise[argmax]

        self.relerror = self.calc_uncert(self.deltanoise)
        self.adjusted_structure = self.structure + (self.structure * self.relerror)
        # Find the first and last index to be at or above maxstructure
        indices = np.where(self.adjusted_structure >= self.maxstructure)[0]
        if len(indices) < 2:
            warnings.warn(
                "no valid range found for adjusted structure parameter; no error estimate"
            )
            return (self.optdm, None, None, None, None)

        # Warn if there are gaps
        if any(np.diff(indices) > 1):
            warnings.warn(
                "gaps in the min-max structure range, where the adjusted structure dips below the maximum structure"
            )
        if indices[0] == 0:
            warnings.warn(
                "minimum dm estimate is at the lower edge of the input dm range"
            )
        if indices[-1] == len(self.dms) - 1:
            warnings.warn(
                "minimum dm estimate is at the lower edge of the input dm range"
            )

        self.mindm = self.dms[indices[0]]
        self.maxdm = self.dms[indices[-1]]
        lowdm = self.mindm - self.optdm
        highdm = self.maxdm - self.optdm
        return (self.optdm, lowdm, highdm, self.mindm, self.maxdm)

    def calc_kc(self, window: int = 5) -> int:
        """Obtain best estimate for the spectral cutoff parameter k_c

        Parameters
        ----------

        window:
            window size for the "rolling average"

        Returns
        -------
        Best estimate for kc


        Raises
        ------
        A ValueError is raised if no k_c value can be found

        Description
        -----------

        Estimates k_c by finding where the noise flattens, and then
        finding the point (index) where the windowed-average of the
        signal dips below the noise.

        See section 4.3 in Sutinjo et al.

        """

        if window < 1:
            raise ValueError("`window` should be a positive integer")

        margin = self.dctdata.shape[1] // 2
        logger.info("noise margin: %d", margin)

        dctdatamax = np.abs(self.dctdata).max(axis=0)

        top = dctdatamax[margin:].mean()
        logger.info("noise flattening level: %f", top)

        self.kc = -1
        n = len(dctdatamax)
        for i in range(window, n):
            mean = np.mean(dctdatamax[i - window : i + 1])
            if mean <= top:  # Signal is at or below the noise
                self.kc = i
                logger.info("found kc: %d", self.kc)
                break

        if self.kc < 0:
            raise ValueError("Failed to find a value for k_c")

        return self.kc

    def calc_uncert(self, noise):
        """Determine the uncertainties for the data

        Following eq 20, Sutinjo et al 2023.

        """

        doublefiltered = self.bandpass @ self.dctdata_lp
        normdoublefiltered = np.linalg.norm(doublefiltered, axis=0)

        dctnoise = dct(noise, norm="ortho")
        dctnoisefiltered = self.bandpass @ dctnoise.T
        normeddctnoise = np.linalg.norm(dctnoisefiltered, axis=0)

        self.error = normeddctnoise / normdoublefiltered

        return self.error

    def plot_spectrum(
        self,
        ax: Axes | None = None,
        grid: bool = True,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plot the spectrum of the (bowtie) data

        Plots dct(I) versus k on a log-log plot.

        If `kc` was calculated or given, plot a vertical line at the `kc` value.

        """

        fig, ax = ensure_figure(ax)

        ax.loglog(np.abs(self.dctdata.T), ".", color="black")
        defaults = {"title": "spectrum", "xlabel": r"$k$", "ylabel": r"$ C^T \cdot i$"}
        set_title_labels(kwargs, defaults, ax)
        if self.kc:
            ax.axvline(x=self.kc)
        if grid:
            ax.grid(color="k", linestyle="--", linewidth=0.5)

        return (fig, ax)

    def plot_structure(self, ax: Axes | None = None, grid: bool = True, **kwargs):
        """Plot the structure parameter versus DM"""

        fig, ax = ensure_figure(ax)

        ax.plot(self.dms, self.structure)
        defaults = {"title": "structure parameter", "xlabel": r"DM"}
        set_title_labels(kwargs, defaults, ax)
        if grid:
            ax.grid(color="k", linestyle="--", linewidth=0.5)

        return (fig, ax)

    def plot_adjusted_structure(
        self, ax: Axes | None = None, grid: bool = True, **kwargs
    ):
        """Plot the adjusted structure parameter versus DM"""

        fig, ax = ensure_figure(ax)

        ax.plot(self.dms, self.adjusted_structure)
        defaults = {"title": "adjusted structure parameter", "xlabel": r"DM"}
        set_title_labels(kwargs, defaults, ax)
        if grid:
            ax.grid(color="k", linestyle="--", linewidth=0.5)

        return (fig, ax)

    def plot_uncertainty(self, ax: Axes | None = None, grid: bool = True, **kwargs):
        """Plot the (relative) uncertainties versus DM"""

        fig, ax = ensure_figure(ax)

        ax.plot(self.dms, self.relerror)
        defaults = {"title": "uncertainty", "xlabel": r"DM"}
        set_title_labels(kwargs, defaults, ax)
        if grid:
            ax.grid(color="k", linestyle="--", linewidth=0.5)

        return (fig, ax)

    def plot_detrended_noise(self, ax: Axes | None = None, **kwargs):
        """Plot the detrended noise"""

        fig, ax = ensure_figure(ax)
        extent = [0, self.nsamples, self.dms[0].value, self.dms[-1].value]
        image = ax.imshow(
            self.noise,
            extent=extent,
            aspect="auto",
            origin="lower",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        ax.figure.colorbar(image, cax=cax, orientation="vertical")
        defaults = {
            "xlabel": "Samples",
            "ylabel": r"$\Delta$DM $\text{pc cm}^{3}$",
            "title": "Detrendend noise",
        }
        set_title_labels(kwargs, defaults, ax)

        return (fig, ax)

    def plot_relative_detrended_noise(self, ax: Axes | None = None, **kwargs):
        """Plot the relative detrended noise"""

        fig, ax = ensure_figure(ax)
        extent = [0, self.nsamples, self.dms[0].value, self.dms[-1].value]
        image = ax.imshow(
            self.deltanoise,
            extent=extent,
            aspect="auto",
            origin="lower",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        ax.figure.colorbar(image, cax=cax, orientation="vertical")
        defaults = {
            "xlabel": "Samples",
            "ylabel": r"$\Delta$DM $\text{pc cm}^{3}$",
            "title": "Relative detrendend noise",
        }
        set_title_labels(kwargs, defaults, ax)

        return (fig, ax)
