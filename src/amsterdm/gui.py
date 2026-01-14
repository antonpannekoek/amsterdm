from bokeh.models import PrintfTickFormatter, LinearAxis, Range1d, ColorBar
import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import rasterize
import hvplot
import hvplot.xarray  # noqa: F401
import numpy as np
import panel as pn
import param
import xarray as xr

from .candidate import openfile
from .utils import symlog10


hv.extension("bokeh")
pn.extension()
hv.config.image_rtol = 1e-1


SOD = 60 * 60 * 24
COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "fire",
    "gray",
]
DM_ZOOM_LEVELS = {
    "Zoom 1x": {"zoom": 1, "step": 5},
    "Zoom 10x": {"zoom": 10, "step": 1},
    "Zoom 100x": {"zoom": 100, "step": 0.1},
    "Zoom 1000x": {"zoom": 1000, "step": 0.01},
}


class CandidatePlot(param.Parameterized):
    dm = param.Number(default=0.0, bounds=(0.0, 2000.0), step=0.1, label="DM")
    dm_zoom = param.Selector(
        objects=DM_ZOOM_LEVELS.keys(), label="DM slider zoom factor"
    )
    colormap = param.Selector(objects=COLORMAPS, default="viridis", label="Colormap")
    cmin = param.Number(default=0.1, bounds=(0.0, 1.0), label="Lower fraction")
    cmax = param.Number(default=0.9, bounds=(0.0, 1.0), label="Upper fraction")
    logimage = param.Boolean(default=False, label="Logarithmic color scale")
    loglc = param.Boolean(default=False, label="Logarithmic light curve y-axis")
    trunclc = param.Number(default=0, label="Lower y limit")
    # datarange = param.Range((0.3, 0.6), bounds=(0, 1))
    badchanlist = param.String(default="", label="Bad channels")
    # Simple flag for cases where the plot method needs to be explicitly triggered
    # update_data = param.Boolean(default=False)
    update_plot = param.Integer(default=0)
    sample_start = param.Integer(default=0, label="Data sample start")
    sample_end = param.Integer(default=0, label="Data sample end")
    sample_reset = param.Action(label="Reset range")
    bkg_left = param.Integer(default=0, label="Left limit")
    bkg_right = param.Integer(default=0, label="Right limit")
    bkg_reset = param.Action(label="Reset range")

    def __init__(self, candidate, invert=True, width=800, **kwargs):
        super().__init__(**kwargs)
        self.candidate = candidate
        self.tap = streams.Tap(x=1, y=1)
        self.tap.param.watch(self._on_tap, ["y"])
        self.invert = invert
        self.width = width
        self.range_stream = hv.streams.RangeX()

        self.ntotal = self.candidate.data.shape[0]
        # Set the sample end based on the actual data
        # (since this can't be done in the class definition)
        self.sample_end = self.ntotal
        self.param.sample_end.default = self.sample_end
        self.sample_reset = self._reset_sample_range

        # Set the background left and right cutoffs based on the actual data
        self.bkg_left = self.ntotal // 3
        self.param.bkg_left.default = self.bkg_left
        self.bkg_right = 2 * self.ntotal // 3
        self.param.bkg_right.default = self.bkg_right
        self.bkg_reset = self._reset_bkg_range

        # Use a range stream to keep track of the zoom factor
        self.range_stream = hv.streams.RangeXY()

        # internal attribute to keep track of manual selection of the sample range
        self._x_viewrange = (None, None)

        self.param.sample_end.bounds = (1, self.candidate.data.shape[0])
        # self.param.sample_reset.default = self._reset_range

        self.dm_slider = pn.Param(
            self.param.dm, widgets={"dm": pn.widgets.FloatSlider}
        )[0]

        self._init_data()

        self.param.watch(
            self._update_data,
            [
                "dm",
                # "datarange",
                "badchanlist",
                # "update_plot",
                "logimage",
                "loglc",
                "trunclc",
                "sample_start",
                "sample_end",
                "bkg_left",
                "bkg_right",
            ],
        )
        self.param.watch(self._update_dm_slider, ["dm_zoom"])

    def _on_tap(self, event):
        # Simply read the current values from the tap stream
        channel = int(self.tap.y + 0.5)
        if channel in self.badchannels:
            self.badchannels.remove(channel)
        else:
            self.badchannels.add(channel)
        self.badchanlist = ",".join(str(value) for value in sorted(self.badchannels))

        self.update_data = True

    def _update_dm_slider(self, event):
        """Update the DM slider range based on the DM slider zoom selector"""
        config = DM_ZOOM_LEVELS[event.new]
        zoom = config["zoom"]
        step = config["step"]
        value = self.dm
        if zoom == 1:
            start = config["start"]
            end = config["end"]
            width = end - start
        else:
            width = 1000 / config["zoom"]
            start = max(0, self.dm - 2 * width)  # ensure no negative values
            end = self.dm + 2 * width
        value = self.dm // step * step
        self.dm_slider.start = start
        self.dm_slider.end = end
        self.dm_slider.step = step
        self.dm_slider.value = value

    def _reset_bkg_range(self, _):
        self.param.update(
            bkg_left=self.param.bkg_left.default,
            bkg_right=self.param.bkg_right.default,
        )

        self._calc_data()

    def _reset_sample_range(self, _):
        self.param.update(
            sample_start=self.param.sample_start.default,
            sample_end=self.param.sample_end.default,
        )

        self._calc_data()

    def add_physical_axes(self, plot, element):
        times = self.candidate.times - self.candidate.times[0]
        times *= SOD
        freqs = self.candidate.freqs
        figure = plot.state
        if "time" not in figure.extra_x_ranges:
            rangex = Range1d(
                start=times[0],
                end=times[-1],
            )
            figure.extra_x_ranges = {"time": rangex}
            figure.add_layout(
                LinearAxis(
                    x_range_name="time",
                    axis_label="time (ms)",
                ),
                "above",
            )
        else:
            # The `rasterize()` call overwrites the secondary x-axis
            # label, so we may have to add it again
            for layout in figure.above:
                print(layout, layout.x_range_name, layout.axis_label)
                if hasattr(layout, "x_range_name") and layout.x_range_name == "time":
                    layout.axis_label = "time [s]"
        if "freq" not in figure.extra_y_ranges:
            rangey = Range1d(
                start=freqs[0],
                end=freqs[-1],
            )
            figure.extra_y_ranges = {"freq": rangey}
            figure.add_layout(
                LinearAxis(
                    y_range_name="freq",
                    axis_label="frequency (MHz)",
                ),
                "right",
            )

    def move_colorbar(self, plot, element):
        """Move the colorbar after (to the right of) the secondary y-axis"""
        figure = plot.state
        # Grab the existing colorbar
        colorbars = [r for r in figure.right if isinstance(r, ColorBar)]
        if not colorbars:
            return
        for colorbar in colorbars:
            figure.right.remove(colorbar)
            figure.add_layout(colorbar, "right")

    def _init_data(self):
        self.badchannels = set()
        self.channels = list(range(1, len(self.candidate.freqs) + 1))
        if self.invert:
            self.channels = self.channels[::-1]
        self.dt = (self.candidate.times - self.candidate.times[0]) * 1000

        self._calc_data()

    def _calc_data(self):
        samplerange = slice(self.sample_start, self.sample_end)
        datarange = (self.bkg_left / self.ntotal, self.bkg_right / self.ntotal)
        self.stokesI, self.bkg = self.candidate.calc_intensity(
            self.dm,
            {128 - value for value in self.badchannels},
            datarange,
            samplerange=samplerange,
            bkg_extra=True,
        )

        self.stokesI = np.ma.filled(
            self.stokesI, np.nan
        )  # Replace the (temporary) mask with NaNs for plotting purposes and `nanpercentile`

        self.lc = np.nansum(self.stokesI, axis=1)
        if self.loglc:
            self.lc = symlog10(self.lc)
        self.lc[self.lc < self.trunclc] = np.nan

        if self.logimage:
            self.stokesI = symlog10(self.stokesI)

    def _update_data(self, *args, **kwargs):
        if self.badchanlist:
            self.badchannels = {int(value) for value in self.badchanlist.split(",")}
        else:
            self.badchannels = set()
        self._calc_data()

        self.update_plot += 1

    @param.depends("update_plot")
    def plot_lc(self, x_range=None, y_range=None):
        samples = np.arange(self.sample_start, self.sample_end)
        lcplot = hv.Curve((samples, self.lc), "samples", "I").opts(
            width=self.width, framewise=True
        )
        if x_range:
            lcplot = lcplot.opts(xlim=x_range)

        return lcplot

    @param.depends("update_plot", "cmin", "cmax", "colormap")
    def plot_waterfall(self, x_range=None, y_range=None):
        samples = np.arange(self.sample_start, self.sample_end)
        ds = xr.Dataset(
            {"data": (["samples", "channel"], self.stokesI)},
            coords={"channel": self.channels, "samples": samples},
        )
        vmin, vmax = np.nanpercentile(self.stokesI, (self.cmin * 100, self.cmax * 100))
        clim = (vmin, vmax)

        waterfallplot = (
            ds.hvplot(
                colorbar=True,
                clim=clim,
                cmap=self.colormap,
                # Turn off explicitly here; use at last step, on the
                # dynamic map (see below)
                rasterize=False,
                # Similarly, explicitly set this at the last step only
                dynamic=False,
            )
            .redim(x="samples")
            .opts(
                width=self.width,
                height=int(self.width / 1.5),
                # Ensure the colorbar width doesn't change by
                # keeping the numbers on the scale fixed
                colorbar_opts={"formatter": PrintfTickFormatter(format="%.2f")},
            )
        )
        bkg_left = hv.VSpan(0, self.bkg_left).opts(
            color="gray",
            alpha=0.4,  # 0.0 is invisible, 1.0 is opaque
            line_width=0,  # Removes the border line if preferred
        )
        bkg_right = hv.VSpan(self.bkg_right, self.ntotal).opts(
            color="gray",
            alpha=0.4,  # 0.0 is invisible, 1.0 is opaque
            line_width=0,  # Removes the border line if preferred
        )
        background_area = bkg_left * bkg_right
        waterfallplot = waterfallplot * background_area

        if x_range:
            waterfallplot = waterfallplot.opts(
                xlim=x_range,
            )

        return waterfallplot

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
        dm_input = pn.Param(
            self.param.dm, widgets={"dm": {"type": pn.widgets.FloatInput, "width": 150}}
        )[0]
        dm_zoom = pn.Param(
            self.param.dm_zoom,
            widgets={"dm_zoom": {"type": pn.widgets.Select, "width": 150}},
        )[0]
        cmin = pn.Param(self.param.cmin, widgets={"cmin": pn.widgets.FloatInput})[0]
        cmax = pn.Param(self.param.cmax, widgets={"cmax": pn.widgets.FloatInput})[0]
        sample_start = pn.Param(
            self.param.sample_start,
            widgets={"sample_start": {"type": pn.widgets.IntInput, "width": 100}},
        )
        sample_end = pn.Param(
            self.param.sample_end,
            widgets={"sample_end": {"type": pn.widgets.IntInput, "width": 100}},
        )
        samplerange = pn.Row(sample_start, sample_end, self.param.sample_reset)

        bkg_left = pn.Param(
            self.param.bkg_left,
            widgets={"bkg_left": {"type": pn.widgets.IntInput, "width": 100}},
        )
        bkg_right = pn.Param(
            self.param.bkg_right,
            widgets={"bkg_right": {"type": pn.widgets.IntInput, "width": 100}},
        )
        bkgrange = pn.Row(bkg_left, bkg_right, self.param.bkg_reset)

        badchanlist = pn.Param(
            self.param.badchanlist,
            widgets={
                "badchanlist": {
                    "type": pn.widgets.TextInput,
                    "placeholder": "comma-separated integers",
                }
            },
        )[0]

        bkgplots = pn.Card(
            self.plot_bkg,
            header=pn.pane.HTML(
                '<h4 style="margin: 0; padding: 0; text-align: left">Background pre<br>bandpass-correction</h4>'
            ),
            collapsed=True,
        )
        # Create a dynamic map and rasterize only here.
        # Otherwise, the re-rasterization and resolution increase upon
        # zooming in are lost.
        # The hooks also need to be moved here, due to some oddity with
        # the secondary x-axis that would cause its label to be overwritten
        # with that of the primary x-axis
        waterfallplot = hv.DynamicMap(self.plot_waterfall, streams=[self.range_stream])
        waterfallplot = rasterize(waterfallplot, dynamic=True).opts(
            hooks=[self.add_physical_axes, self.move_colorbar]
        )

        self.tap.source = waterfallplot

        lcplot = hv.DynamicMap(self.plot_lc, streams=[self.range_stream])
        plots = pn.Column(lcplot, pn.Row(bkgplots, waterfallplot))
        trunclc = pn.Param(
            self.param.trunclc,
            widgets={"trunclc": {"type": pn.widgets.FloatInput, "width": 100}},
        )[0]

        lcsettings = pn.Card(
            pn.Row(
                self.param.loglc,
                trunclc,
            ),
            title="Light curve settings",
            collapsed=True,
        )
        colorsettings = pn.Card(
            pn.Column(
                self.param.logimage,
                self.param.colormap,
                pn.Row(cmin, cmax),
            ),
            title="Colormap settings",
            collapsed=True,
        )

        data = pn.Card(
            pn.Column(
                samplerange,
                badchanlist,
            ),
            title="Data & bad channels",
            collapsed=False,
        )
        background = pn.Card(
            pn.Column(
                bkgrange,
            ),
            title="Background",
            collapsed=False,
        )
        dmsettings = pn.Card(
            pn.Row(
                self.dm_slider,
                dm_zoom,
                dm_input,
            ),
            title="Dispersion measure",
            collapsed=False,
        )
        layout = pn.Row(
            pn.Column(
                pn.pane.HTML(
                    f'<h1 style="margin: 0; padding: 0; text-align: center">{self.candidate.filename}</h1>'
                ),
                plots,
            ),
            pn.Column(
                lcsettings,
                colorsettings,
                data,
                background,
                dmsettings,
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
