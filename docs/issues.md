Please file bug reports, feature requests and other issues through [GitHub issues](https://github.com/antonpannekoek/amsterdm/issues).

# Issues and to-dos

- allow core functions to accept two-dimensional arrays of `<samples, channels>`, where the implied polarization dimension is Stokes I.

- consider other orderings of dimensions; in particular for a FITS file (this would be noted in the header with special keywords). This can speed up reading the data, in particular if, for example, some of the polarization channels can be completely ignored: when this dimension is the outer dimension (for FITS ordering, i.e., column major Fortran style), large continuous blocks of data can be easily skipped.

- provide a configuration file with essential information for e.g.
  - additional plotting information
  - extra information for fields in the GUI
  - setting the channel <-> frequency conversion correctly (e.g., whether `fch1` refers to the top, bottom or mid of the first channel)
  - additional information stored in a FITS file (through extra keywords)

  This configuration file would be a TOML file, which is a straightforward (e.g., not YAML), readable (e.g., not JSON) and strict format with a reasonable amount of data types (date-time, lists, dicts; e.g., not a configparser ini file).

- Render the documentation with Sphinx. This will automatically extract the API documentation from the doc-strings. Consider whether to change current documentation to reStructured Text (reST); this may have more flexibility and options (such as admonitions), but is slightly less readable and requires a bit more work when writing (note that GitHub renders reST as well as Markdown).

  The MyST parser can use Markdown for the normal documentation; ideally, the doc-strings remain reST, in NumPy style.
