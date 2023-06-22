"""
GLCC

Geographic Projection Parameters

The data dimensions of the Geographic projection for the global
land cover characteristics data set are 21,600 lines (rows) and
43,200 samples (columns) resulting in a data set size of
approximately 933 megabytes for 8-bit (byte) images.
The following is a summary of the map projection parameters
used for the Geographic projection:

Projection Type: Geographic

    Units of measure: arc seconds
    Pixel Size: 30 arc seconds
    Radius of sphere: 6370997 m.
    XY corner coordinates (center of pixel) in projection units (arc seconds):
        Lower left: (-647985, -323985)
        Upper left: (-647985, 323985)
        Upper right: (647985, 323985)
        Lower right: (647985, -323985)

Please Note: The geographic projection is not an equal-area projection,
in contrast with the Interrupted Goode Homolosine projection.
Users should be advised that area statistics calculated from data in
the Geographic projection will not be correct because of the areal
distortion inherent to this projection.


Interrupted Goode Homolosine Projection Parameters

The data dimensions of the Interrupted Goode Homolosine projection
for the global land cover characteristics data set are 17,347
lines (rows) and 40,031 samples (columns) resulting in a data set
size of approximately 695 megabytes for 8-bit (byte) images.
The following is a summary of the map projection parameters used
for the Interrupted Goode Homolosine projection:

Projection Type: Interrupted Goode Homolosine

    Units of measure: meters
    Pixel Size: 1000 meters
    Radius of sphere: 6370997 m.
    XY corner coordinates (center of pixel) in projection units (meters):
        Lower left: (-20015000, -8673000)
        Upper left: (-20015000, 8673000)
        Upper right: (20015000, 8673000)
        Lower right: (20015000, -8673000)

"""


import cf_xarray as cfxr  # noqa
import numpy as np
import rasterio
import xarray as xr

from .transform import transform_bounds, transform_yx
from .utils import glcc_grid


class OlsonGlobalEcosystem:
    classes = {
        0: "INTERRUPTED AREAS (GLOBAL GOODES HOMOLOSINE PROJECTION)",
        1: "URBAN",
        2: "LOW SPARSE GRASSLAND",
        3: "CONIFEROUS FOREST",
        4: "DECIDUOUS CONIFER FOREST",
        5: "DECIDUOUS BROADLEAF FOREST",
        6: "EVERGREEN BROADLEAF FORESTS",
        7: "TALL GRASSES AND SHRUBS",
        8: "BARE DESERT",
        9: "UPLAND TUNDRA",
        10: "IRRIGATED GRASSLAND",
        11: "SEMI DESERT",
        12: "GLACIER ICE",
        13: "WOODED WET SWAMP",
        14: "INLAND WATER",
        15: "SEA WATER",
        16: "SHRUB EVERGREEN",
        17: "SHRUB DECIDUOUS",
        18: "MIXED FOREST AND FIELD",
        19: "EVERGREEN FOREST AND FIELDS",
        20: "COOL RAIN FOREST",
        21: "CONIFER BOREAL FOREST",
        22: "COOL CONIFER FOREST",
        23: "COOL MIXED FOREST",
        24: "MIXED FOREST",
        25: "COOL BROADLEAF FOREST",
        26: "DECIDUOUS BROADLEAF FOREST",
        27: "CONIFER FOREST",
        28: "MONTANE TROPICAL FORESTS",
        29: "SEASONAL TROPICAL FOREST",
        30: "COOL CROPS AND TOWNS",
        31: "CROPS AND TOWN",
        32: "DRY TROPICAL WOODS",
        33: "TROPICAL RAINFOREST",
        34: "TROPICAL DEGRADED FOREST",
        35: "CORN AND BEANS CROPLAND",
        36: "RICE PADDY AND FIELD",
        37: "HOT IRRIGATED CROPLAND",
        38: "COOL IRRIGATED CROPLAND",
        39: "COLD IRRIGATED CROPLAND",
        40: "COOL GRASSES AND SHRUBS",
        41: "HOT AND MILD GRASSES AND SHRUBS",
        42: "COLD GRASSLAND",
        43: "SAVANNA (WOODS)",
        44: "MIRE, BOG, FEN",
        45: "MARSH WETLAND",
        46: "MEDITERRANEAN SCRUB",
        47: "DRY WOODY SCRUB",
        48: "DRY EVERGREEN WOODS",
        49: "VOLCANIC ROCK",
        50: "SAND DESERT",
        51: "SEMI DESERT SHRUBS",
        52: "SEMI DESERT SAGE",
        53: "BARREN TUNDRA",
        54: "COOL SOUTHERN HEMISPHERE MIXED FORESTS",
        55: "COOL FIELDS AND WOODS",
        56: "FOREST AND FIELD",
        57: "COOL FOREST AND FIELD",
        58: "FIELDS AND WOODY SAVANNA",
        59: "SUCCULENT AND THORN SCRUB",
        60: "SMALL LEAF MIXED WOODS",
        61: "DECIDUOUS AND MIXED BOREAL FOREST",
        62: "NARROW CONIFERS",
        63: "WOODED TUNDRA",
        64: "HEATH SCRUB",
        65: "COASTAL WETLAND - NW",
        66: "COASTAL WETLAND - NE",
        67: "COASTAL WETLAND - SE",
        68: "COASTAL WETLAND - SW",
        69: "POLAR AND ALPINE DESERT",
        70: "GLACIER ROCK",
        71: "SALT PLAYAS",
        72: "MANGROVE",
        73: "WATER AND ISLAND FRINGE",
        74: "LAND, WATER, AND SHORE",
        75: "LAND AND WATER, RIVERS",
        76: "CROP AND WATER MIXTURES",
        77: "SOUTHERN HEMISPHERE CONIFERS",
        78: "SOUTHERN HEMISPHERE MIXED FOREST",
        79: "WET SCLEROPHYLIC FOREST",
        80: "COASTLINE FRINGE",
        81: "BEACHES AND DUNES",
        82: "SPARSE DUNES AND RIDGES",
        83: "BARE COASTAL DUNES",
        84: "RESIDUAL DUNES AND BEACHES",
        85: "COMPOUND COASTLINES",
        86: "ROCKY CLIFFS AND SLOPES",
        87: "SANDY GRASSLAND AND SHRUBS",
        88: "BAMBOO",
        89: "MOIST EUCALYPTUS",
        90: "RAIN GREEN TROPICAL FOREST",
        91: "WOODY SAVANNA",
        92: "BROADLEAF CROPS",
        93: "GRASS CROPS",
        94: "CROPS, GRASS, SHRUBS",
        95: "EVERGREEN TREE CROP",
        96: "DECIDUOUS TREE CROP",
        100: "NO DATA",
    }


def create_dataset(tif=None, coords=True, bounds=True):
    """Creates an xarray dataset from GLCC geotif."""
    if tif is None:
        tif = "/work/ch0636/g300046/glcc/glccgbg20_tif/gbogeg20.tif"

    print(f"reading: {tif}")
    with rasterio.open(tif) as img:
        data = img.read()
        crs = img.crs

    print("creating dataset")
    ds = to_dataset(np.squeeze(data))
    crs = crs.to_proj4()
    ds.attrs["proj4"] = crs

    if coords is True:
        print(f"creating coordinates from crs: {crs}")
        lat, lon = transform_yx(ds.y, ds.x, crs)
        ds = ds.assign_coords(lon=lon, lat=lat)

    if bounds is True:
        print(f"creating bounds from crs: {crs}")
        ds = ds.cf.add_bounds(("y", "x"))
        yv, xv = transform_bounds(ds.y_bounds, ds.x_bounds, crs)
        ds = ds.assign_coords(
            lon_bounds=xv.transpose(..., "vertices"),
            lat_bounds=yv.transpose(..., "vertices"),
        )

    return ds


def to_dataset(data):
    # create grid
    x = np.linspace(*glcc_grid.x)
    y = np.linspace(*glcc_grid.y)

    data = np.flipud(data)

    ds = xr.Dataset(
        data_vars=dict(
            glcc=(["y", "x"], data),
        ),
        coords=dict(
            x=(["x"], x),
            y=(["y"], y),
            index=(["index"], list(OlsonGlobalEcosystem.classes.keys())),
            type=(["index"], list(OlsonGlobalEcosystem.classes.values())),
        ),
        attrs=dict(
            description="Global Land Cover Characteristics Data Base Version 2.0."
        ),
    )
    ds.x.attrs["axis"] = "X"
    ds.y.attrs["axis"] = "Y"
    return ds.chunk(glcc_grid.chunks)


def glcc_legend(url):
    """reads a glcc legend file"""
    with open(url) as f:
        lines = f.readlines()
    name = lines[0].strip()
    classes = {}
    for line in lines[4:]:
        s = line.strip().split(" ", 1)
        classes[int(s[0])] = s[1]
    return name, classes


def plot(da):
    """Plots data in Interrupted Goode Homolosine projection"""
    from cartopy import crs as ccrs
    from matplotlib import pyplot as plt

    transform = ccrs.InterruptedGoodeHomolosine()

    xlocs = range(-180, 180, 10)
    ylocs = range(-90, 90, 5)

    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=transform)

    ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", xlocs=xlocs, ylocs=ylocs
    )
    da.plot(ax=ax, transform=transform)
    ax.coastlines(resolution="110m", color="blue", linewidth=1)

    return plt
