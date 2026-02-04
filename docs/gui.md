# Graphical User Interface (gui)

:::{attention}
The gui is meant for exploratory analysis of a single burst. It is not meant for accurate analysis or creating precisely configured figures, nor can it (currently) handle multiple bursts.
:::

## Starting

You can start the gui from the command line with

```shell
python -m amsterdm.gui b_59881.fil
```

It provides a localhost URL, that you then use in a browser of your choice (it will not open up in a browser automatically).

The command line will have logging output, and sometimes show (lenghty) backtrace, even if no error in the browser window occur; this is in part due to the interaction between several parts that make up the gui framework, and most of the time, this can be ignored without problems.


## Basic interface

The basic interface uses [Bokeh](https://bokeh.org/) for plotting, and the plots come with a small Bokeh navigation panel on the side. These are the following:

- The rainbow-colored diaphragm will open the Bokeh home page (in a new tab); it's otherwise not useful, and may at some point be removed from the panel
- The crossed arrows allow you to pan the image with the mouse (selected by default)
- The dashed square is a box zoom, and allows you to select an area to zoom in to
- The +-loupe with a little mouse is for mouse-wheel zoom (selected by default)
- The download button allows you to save that particular plot
- The reset button resets the complete plot (convenient when you're all zoomed in, or zoomed too far out). Occasionally, this may not work.
- The hover button shows position information as you mouse over the plot. It is on by default, but that may feel distracting.

Note that zooming in with the mouse zooms along both the x and y axes. For zooming just the x-axis (sample / time axis), prefer the box zoom.

You'll find that zooming on one plot, automatically zooms the other plot as well. So to zoom in on just the burst, it may be convenient to zoom in on the light curve plot, and the waterfall plot will follow along.


On the right of the page are various panels with settings. Most of these are fairly obvious, and values have been set to a reasonable default.

The background limit should be in sample values (this is in contrast to the core interface when using the package directly, where it is in fractions between 0 and 1. This sets two background sections: one from zero to "left", and one from "right" to the last sample.

The dispersion measure can be set manually, for a good first estimate. After that, you can set the slider zoom factor higher, and move the slider to change the DM. This is responsive, albeit at a slow rate; the larger the input file, the slower the response. Naturally, very small changes in DM will not show any change if the time resolution of the data is too low, since then the dispersion change is all contained in one pixel width (even at the largest frequency difference).

### Bad channels

Any bad channels can be set in the "Channels" panel, as a comma-separated list of integers.

Alternatively, you can click a row on the waterfall plot to flag that particular channel (this may take a second to register); the "Bad channels" field in the panel will be automatically updated. Click the same row again to deselect it.



# Issues and to-dos

A short list of known issues and future items to implement, specifically for the gui

- there may be an issue with the channel numbers, in particular for observations / telescopes where the frequencies go downward (`foff` is negative) from the reference frequency
- interactively selected bad channels causes a little "hop" in the waterfall plot (minor shrink and stretch); it's unclear what's causing this
- additional fields in the "Channels" panel would contain options for the user to alter the values of the frequency of channel 1, the frequency offset, and the "anchor" point of the frequency in the channel ("mid", "top" or "bottom").
- the "Dispersion measure" panel needs an additional field for the reference frequency (by default, this is the frequency for channel 1)
- the coherence dispersion should also be written in the "Dispersion measure" panel. Since this may be not be available in the header information of the input file, this will be a user-editable field
- figures to be added:
  - signal-to-noise ratio figure
  - bowtie plot
  Both plots take a bit of time to generate, so it is likely that these will be on-demand plots, and not regenerated automatically with any change in DM, color scales or otherwise.
- allow for a selection on the light curve. This can be used for a (peak) signal-to-noise calculation, that includes only one specific peak to optimize for.
