import numpy as np
import xarray as xr
from pyproj import CRS

from .utils import glcc_grid


def add_lat_lon(ds, crs):
    """add latitude and longitude coordinates to ds"""
    lat, lon = transform_xy(ds, crs)
    return ds.assign_coords(
        lat=lat.where(~np.isinf(lat)), lon=lon.where(~np.isinf(lon))
    )


def transform_xy(ds, crs):
    """transform x and y coordinates of ds to ESPG:4326"""
    from pyproj import Transformer

    crs = CRS(crs)
    world = CRS("EPSG:4326")
    transformer = Transformer.from_crs(crs, world)

    y_stack, x_stack = xr.broadcast(ds.y, ds.x)
    y_stack = y_stack.chunk(glcc_grid.chunks)
    x_stack = x_stack.chunk(glcc_grid.chunks)

    core_dims = 2 * [[]]

    yt, xt = xr.apply_ufunc(
        transformer.transform,  # first the function
        x_stack,  # now arguments in the order expected by 'interp1_np'
        y_stack,
        output_core_dims=core_dims,
        dask="parallelized",
        kwargs={},
    )

    return yt, xt
