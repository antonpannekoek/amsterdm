from bokeh.models import PrintfTickFormatter
import holoviews as hv
from holoviews import streams
import hvplot
import hvplot.xarray  # noqa: F401
import numpy as np
import panel as pn
import param
import xarray as xr

from .candidate import openfile


hv.extension("bokeh")
pn.extension()
hv.config.image_rtol = 1e-1


COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "fire",
    "gray",
]


class CandidatePlot(param.Parameterized):
    dm = param.Number(default=0.0, bounds=(0.0, 2000.0), step=0.1, label="DM")
    # image_y = param.Number(default=1, allow_None=True)
    colormap = param.Selector(objects=COLORMAPS, default="viridis", label="Colormap")
    cmin = param.Number(default=0.1, bounds=(0.0, 1.0), label="Lower fraction")
    cmax = param.Number(default=0.9, bounds=(0.0, 1.0), label="Upper fraction")
    objrange = param.Range((0.3, 0.6), bounds=(0, 1))
    badchanlist = param.String(default="", label="Bad channels")
    # Simple flag for cases where the plot method needs to be explicitly triggered
    update_data = param.Boolean(default=False)
    update_plot = param.Integer(default=0)

    def __init__(self, candidate, invert=True, width=800, **kwargs):
        super().__init__(**kwargs)
        self.candidate = candidate
        self.tap = streams.Tap(x=1, y=1)
        self.tap.param.watch(self._on_tap, ["y"])
        self.invert = invert
        self.width = width

        self._init_data()

        self.param.watch(
            self._update_data, ["dm", "objrange", "badchanlist", "update_data"]
        )

    def _on_tap(self, event):
        # Simply read the current values from the tap stream
        channel = int(self.tap.y + 0.5)
        if channel in self.badchannels:
            self.badchannels.remove(channel)
        else:
            self.badchannels.add(channel)
        self.badchanlist = ",".join(str(value) for value in sorted(self.badchannels))

        self.update_data = True

    def _init_data(self):
        self.badchannels = set()
        self.channels = list(range(1, len(self.candidate.freqs) + 1))
        if self.invert:
            self.channels = self.channels[::-1]
        self.dt = (self.candidate.times - self.candidate.times[0]) * 1000

        self._calc_data()

    def _calc_data(self):
        dm = {
            "dm": self.dm,
            "freq": self.candidate.freqs,
            "tsamp": self.candidate.header["tsamp"],
        }
        self.stokesI, self.bkg = self.candidate.calc_intensity(
            {128 - value for value in self.badchannels},
            dm,
            self.objrange,
            ...,
            bkg_extra=True,
        )

        self.stokesI = np.ma.filled(
            self.stokesI, np.nan
        )  # Replace the (temporary) mask with NaNs for plotting purposes and `nanpercentile`

    def _update_data(self, event):
        self.update_data = (
            False  # change back, so assignment in _on_tap triggers this method
        )

        if self.badchanlist:
            self.badchannels = {int(value) for value in self.badchanlist.split(",")}
        else:
            self.badchannels = set()

        self._calc_data()

        self.update_plot += 1

    @param.depends("update_plot")
    def plot_lc(self):
        lc = np.nansum(self.stokesI, axis=1)
        lcplot = hv.Curve((self.dt, lc), "t", "I").opts(width=self.width)

        return lcplot

    @param.depends("update_plot", "cmin", "cmax", "colormap")
    def plot_image(self):
        ds = xr.Dataset(
            {"data": (["t", "channel"], self.stokesI)},
            coords={"channel": self.channels, "t": self.dt},
        )
        vmin, vmax = np.nanpercentile(self.stokesI, (self.cmin * 100, self.cmax * 100))
        clim = (vmin, vmax)

        dynspec = (
            ds.hvplot(
                colorbar=True,
                clim=clim,
                cmap=self.colormap,
                rasterize=True,
            )
            .redim(x="t")
            .opts(
                width=self.width,
                height=int(self.width / 1.5),
                # Ensure the colorbar width doesn't change by
                # keeping the numbers on the scale fixed
                colorbar_opts={"formatter": PrintfTickFormatter(format="%.2f")},
            )
        )
        self.tap.source = dynspec

        return dynspec

    @param.depends("update_plot")
    def plot_bkg(self):
        bkg_lc_mean = hv.Curve(
            (self.bkg["mean"], self.channels), "mean", "channel"
        ).opts(width=200, height=int(self.width / 1.5))
        bkg_lc_std = hv.Curve((self.bkg["std"], self.channels), "std", "channel").opts(
            width=200, height=int(self.width / 1.5)
        )

        return pn.Row(bkg_lc_mean, bkg_lc_std)

    def panel(self):
        slider = pn.Param(self.param.dm, widgets={"dm": pn.widgets.FloatSlider})[0]
        input_widget = pn.Param(self.param.dm, widgets={"dm": pn.widgets.FloatInput})[0]
        cmin = pn.Param(self.param.cmin, widgets={"cmin": pn.widgets.FloatInput})[0]
        cmax = pn.Param(self.param.cmax, widgets={"cmax": pn.widgets.FloatInput})[0]
        badchanlist = pn.Param(
            self.param.badchanlist,
            widgets={
                "badchanlist": {
                    "type": pn.widgets.TextInput,
                    "placeholder": "comma-separated integers",
                }
            },
        )[0]
        objrange = pn.Param(
            self.param.objrange, widgets={"objrange": pn.widgets.RangeSlider}
        )[0]
        objrange.width = self.width
        objrange.step = 0.01

        bkgplots = pn.Card(
            self.plot_bkg,
            header=pn.pane.HTML(
                '<h4 style="margin: 0; padding: 0; text-align: left">Background pre<br>bandpass-correction</h4>'
            ),
            collapsed=True,
        )
        image_plot = pn.Column(self.plot_image, objrange)
        plots = pn.Column(self.plot_lc, pn.Row(bkgplots, image_plot))

        layout = pn.Row(
            pn.Column(
                plots,
            ),
            pn.Column(
                pn.Row(slider, input_widget),
                self.param.colormap,
                pn.Row(cmin, cmax),
                badchanlist,
            ),
        )
        return layout


def main(filename):
    with openfile(filename) as candidate:
        plot = CandidatePlot(candidate, invert=True, width=1200)
        layout = plot.panel()

        pn.serve(
            layout,
            title="FRB Candidate - interactive DM",
            port=5006,
            show=False,
            autoreload=True,  # This is equivalent to --dev
            dev=True,  # Enable development mode (includes autoreload + debug features)
            num_procs=1,  # Use single process (better for debugging)
        )


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("file", help="Filterbank file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.file)
