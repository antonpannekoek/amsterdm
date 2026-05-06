from astropy import units


DEFAULT_BACKGROUND_RANGE = [(0, 0.333), (0.666, 1)]

# Unit to express the DM value in
DMUNIT = units.cm**-3 * units.pc

# According to Nimmo et al 2022, 10.1038/s41550-021-01569-9,
# this is also the value that is used in dspsr (digifil) and SFXC
# Units: MHz^2 cm^3 pc^-1 ms
DMCONST = 1.0 / 2.41e-4 * units.MHz**2 * units.s / DMUNIT

# Seconds in a day
SOD = 60 * 60 * 24 * units.second
