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

But before installing the package, you may want to create an (empty) virtual environment first and activate it, for example with

```
python -m venv amsterdm
source amsterdm/bin/activate
```

but a [`uv`](https://docs.astral.sh/uv/pip/environments/), [`Mamba`](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) or [`Conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) virtual environment can also work. The latter tools can also install a specific Python version for you.

## Development installation

If you want to do development on the package, it is best to fork the package from its GitHub repository, then clone your local fork to your work machine, and install from the clone. After setting up and activating a virtual environment, install an editable version of the package directly from your local repository, with some extra packages as well:

```
pip install -e '.[dev,extra]'
```

The `dev` dependencies include [`ruff`](https://docs.astral.sh/ruff/), which is used for formatting and linting of the code, and [`pytest`](https://docs.pytest.org/en/stable/) for running unit tests. The `extra` package contains the [Jupyter](https://jupyter.org/) suite of tools (the terminal, notebook and lab interfaces).

# Running

For a quick test run with the interactive gui, run the following command on the command line:
```
python -m amsterdm.gui <filterbank-file>
```

This will open the gui in a new tab in your default browser, with the filterbank file opened and plotted in the diagnostic plots.

# License

This software is copyright 2025, University of Amsterdam, and is distributed under the MIT license.
