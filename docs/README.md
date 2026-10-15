This directory contains documentation for the AmsterDM package.

See the [`index.md`](index.md) file for a table of contents.

To create formatted documentation using Sphinx, ensure you have Sphinx installed with additional packages sphinx-autoapi, sphinx-design, fureo and myst-parser (`pip install '.[docs]'` does this), and then you should be able to run

```shell
make html
```

in this directory. The output will be in `_build/html/`; start with the `index.html` file.

If you like a PDF file, use

```shell
make latexpdf
```

You will need to have LaTeX installed.
