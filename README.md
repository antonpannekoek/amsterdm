# AmsterDM

AmsterDM is a Python package that provides functionality to retrieve the best estimates for the dispersion measure (DM) of fast radio bursts (FRBs), in particularly in cases where the DM may vary across the light curve. This estimation can be done interactively through a gui, with various diagnostic plots that can be updated on the fly.

# Dependencies

AmsterDM depends on a set of Python packages: NumPy, AstroPy, Xarray and the Holoviews ecosystem. It is likely that precompiled binary versions of these packages exist for your platform and Python version, and you use a simple `pip install` to have the packages installed.

The required minimum Python version is 3.11.

The software has been tested on an M4 Macbook Pro, with Firefox 146 or Safari 26 for the gui.

# Installation

A quick installation of the package itself can be done directly from the GitHub repository, with

```
pip install git+https://github.com/antonpannekoek/amsterdm.git
```

This will install the dependencies for you as well.

## Virtual environment

Bbefore installing the package, you may want to create an (empty) virtual environment and activate it, for example with

```
python -m venv amsterdm
source amsterdm/bin/activate
```

A [`uv`](https://docs.astral.sh/uv/pip/environments/), [`Mamba`](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) or [`Conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) virtual environment can also work. The latter tools can also install a specific Python version for you.

## Development installation

If you want to do development on the package, it is best to fork the package on GitHub, then clone your local fork to your work machine, and install from the clone. After setting up and activating a virtual environment, install an editable version of the package directly from your local repository, with some extra packages as well:

```
pip install -e '.[dev,extra]'
```

The `dev` dependencies include [`ruff`](https://docs.astral.sh/ruff/), which is used for formatting and linting of the code, [`pytest`](https://docs.pytest.org/en/stable/) for running unit tests, and [MyST](https://myst-parser.readthedocs.io/en/latest/intro.html) for documentation. The `extra` package contains the [Jupyter](https://jupyter.org/) suite of tools (the terminal, notebook and lab interfaces).

# Running

For a quick test run with the interactive gui, run the following command on the command line:
```
python -m amsterdm.gui <filterbank-file>
```

This will open the gui in a new tab in your default browser, with the filterbank file opened and plotted in the diagnostic plots.

Some more details are in the `docs/gui.md` file.

## Examples

There are a few example scripts in the `examples/` directory:

- `plots.py`: create one or more plots for FRB data. Run with `plots.py --help` or simply by itself, `plots.py`, to see the possible plots and other options. The required input file should be a filterbank or FITS file

- `simulatepy`: create a very simple simulation of FRB data and save to a FITS file


# File formats

At the moment, only Filterbank files and certain FITS files are accepted. It is likely that not all Filterbank files will work; it has only been tested for a limited set of telescopes.

The FITS file is a format introduced for this package: it follows standard FITS conventions, but makes it easier to read the file (any standard FITS utility will work), and the header is more flexible.

FITS extensions make it also possible to either store additional data in other extensions (a proper list of flagged channels could be stored in a FITS table extension; currently, this information is stored in a string as a comma-separated list of integers). Or alternatively, multiple observations of the same burst can be stored in a single FITS file, each in their own extension.

A simple tool is available to convert a Filterbank file to a FITS file. It can be used as

```
python -m amsterdm.tools.fil2fits <filterbank-file> [FITS-file]
```

The FITS file is optional; if not given, the output file name is the same as the input Filterbank file, with the extension changed from `.fil` to `.fits`

# License

This software is copyright 2026, University of Amsterdam, and is distributed under the GPL-3.0 license. See the LICENSE file for the full text.
