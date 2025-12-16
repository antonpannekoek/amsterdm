def coord2deg(coord, factor=1):
    """Convert a Filterbank coordinate format to degrees"""
    sign = -1 if coord < 0 else 1
    degrees, coord = divmod(coord, 10_000)
    minutes, seconds = divmod(coord, 100)
    degrees = degrees + minutes / 60 + seconds / 3600
    return sign * degrees * factor
