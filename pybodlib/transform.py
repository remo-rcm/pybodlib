import numpy as np
import xarray as xr
from pyproj import CRS

from .utils import glcc_grid


def transform_yx(y, x, crs):
    """transform x and y coordinates of ds to ESPG:4326"""
    from pyproj import Transformer

    crs = CRS(crs)
    world = CRS("EPSG:4326")
    transformer = Transformer.from_crs(crs, world)

    y_stack, x_stack = xr.broadcast(y, x)
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

    return ~np.isinf(yt), ~np.isinf(xt)


def transform_bounds(y_bounds, x_bounds, crs):
    # order is counterclockwise starting from lower left vertex
    v1 = transform_yx(
        y_bounds.isel(bounds=0),
        x_bounds.isel(bounds=0),
        crs,
    )
    v2 = transform_yx(y_bounds.isel(bounds=0), x_bounds.isel(bounds=1), crs)
    v3 = transform_yx(y_bounds.isel(bounds=1), x_bounds.isel(bounds=1), crs)
    v4 = transform_yx(y_bounds.isel(bounds=1), x_bounds.isel(bounds=0), crs)

    y_vertices = xr.concat([v1[0], v2[0], v3[0], v4[0]], dim="vertices")
    x_vertices = xr.concat([v1[1], v2[1], v3[1], v4[1]], dim="vertices")

    return y_vertices, x_vertices
