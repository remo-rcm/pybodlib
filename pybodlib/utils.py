import numpy as np


class glcc_grid:
    # 17,347 lines (rows) and 40,031
    ny = 17347
    nx = 40031
    line = 8676
    R = 6370997.0
    chunks = {"x": 4010, "y": 1738}
    x = (-20015000.0, 20015000.0, nx)
    y = (-8673000.0, 8673000.0, ny)


def _tif_to_numpy(url):
    import rasterio

    with rasterio.open(url) as img:
        ds = img.read()
    return ds


def read_img(filename, dtype="uint8"):
    """Read DEM file

    The DEM is provided as 16-bit signed integer data in a simple binary raster.
    There are no header or trailer bytes imbedded in the image.  The data are
    stored in row major order (all the data for row 1, followed by all the data
    for row 2, etc.).

    """
    with open(filename, "r") as f:
        a = np.fromfile(f, dtype=dtype)
    return a
