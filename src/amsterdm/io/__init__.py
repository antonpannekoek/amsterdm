# Use explicit re-exports to make the read_* functions part of the
# general io subpackage namespace

from .core import read_fileformat as read_fileformat
from .filterbank import read_filterbank as read_filterbank
from .fits import read_fits as read_fits
