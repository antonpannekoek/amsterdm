# Development

For development of the package, fork the repository on GitHub, clone your personal fork, and install this as a editable installation in a virtual environment, with the development (and extra) packages:

```
pip install -e '.[dev,extra]'
```

The project uses a single `pyproject.toml` file for all its configuration. Development tools can also be configured through this file. There are no C-extensions (yet), so no `setup.py` or similar is needed.

### Dependencies

Dependencies and required versions are listed in `pyproject.toml`. Note that these use the `~=` specifier, to indicate a so-called [compatible release](https://packaging.python.org/en/latest/specifications/version-specifiers/#compatible-release) (some of these may be loosened a bit).

## Code changes upstream

For bug fixes, new functionality or any code changes, please work on an appropriately named branch (the branch name ends up in the merge commit), and try to keep the branch and corresponding pull request focused to the issue at hand. Make sure the branch is up-to-date with the most recent upstream/main.

It is often good to first open an issue instead of directly submitting a pull request (obvious bug or documentation fixes aside). Sometimes, other people have other (larger) plans for a specific item or feature, and a brief discussion may be needed as to what the best approach is.

### Graphical User Interface (gui)

The gui is built with the holoviews ecosystem. It relies somewhat on the core functions, but some functionality is more practially re-implemented inside the gui module.

There are a lot of hacks for certain interactivity (for example, keeping the zoom level consistent while changing other options such as the colorscale; this wasn't straightforward). Most of these are annoated with comments, but this may make certain parts of the gui somewhat obscure.

Note that a lot of the gui development has been done with the use of LLMs, and as such, may not always be the actual best practice for any of the holoviews pcakages. Also, a lot in holoviews is still changing. This means a lot of functionality and implementation may need to be changed in the near future.


## Formatting and linting

Formatting and linting is done with [Ruff](https://docs.astral.sh/ruff/). This should be done for every commit, and should pass cleanly. This can be checked with an appropriate `pre-commit` hook (the hook can also format the code, but it may be preferred to manually have a last look at the code after formatting, which a hook won't do. It may be preferred to just run a formatting check in the hook).


## Testing

Unit tests can be found in the
Unit tests are created within the Pytest framework, can be run with a simple `pytest` command (the `[tool.pytest.ini_options]` section in `pyproject.toml` isolates the `tests` directory.

The unit tests cover only a few functions.
### Continuous Integration (CI)

The CI tests the following:

- installation of the package
- formatting, with `ruff format --check`
- linting, with `ruff check`
- running the gui module with the `--help` option; this will not attempt to start the gui, but does test the module and imports
- running the unit tests

The Python versions tested are 3.11, 3.12, 3.13 and 3.14.

The details can be found in `.github/workflows/ci.yml`.
