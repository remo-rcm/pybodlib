"""
GLCC

Geographic Projection Parameters

The data dimensions of the Geographic projection for the global land cover characteristics data set are 21,600 lines (rows) and 43,200 samples (columns) resulting in a data set size of approximately 933 megabytes for 8-bit (byte) images. The following is a summary of the map projection parameters used for the Geographic projection: Projection Type: Geographic

    Units of measure: arc seconds
    Pixel Size: 30 arc seconds
    Radius of sphere: 6370997 m.
    XY corner coordinates (center of pixel) in projection units (arc seconds):
        Lower left: (-647985, -323985)
        Upper left: (-647985, 323985)
        Upper right: (647985, 323985)
        Lower right: (647985, -323985) Please Note: The geographic projection is not an equal-area projection, in contrast with the Interrupted Goode Homolosine projection. Users should be advised that area statistics calculated from data in the Geographic projection will not be correct because of the areal distortion inherent to this projection.

Interrupted Goode Homolosine Projection Parameters

The data dimensions of the Interrupted Goode Homolosine projection for the global land cover characteristics data set are 17,347 lines (rows) and 40,031 samples (columns) resulting in a data set size of approximately 695 megabytes for 8-bit (byte) images. The following is a summary of the map projection parameters used for the Interrupted Goode Homolosine projection: Projection Type: Interrupted Goode Homolosine

    Units of measure: meters
    Pixel Size: 1000 meters
    Radius of sphere: 6370997 m.
    XY corner coordinates (center of pixel) in projection units (meters):
        Lower left: (-20015000, -8673000)
        Upper left: (-20015000, 8673000)
        Upper right: (20015000, 8673000)
        Lower right: (20015000, -8673000)
"""


import numpy as np
import xarray as xr


class glcc_grid:
    # 17,347 lines (rows) and 40,031
    ny = 17347
    nx = 40031
    line = 8676
    R = 6370997.0


class OlsonGlobalEcosystemLegend:
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


def read_img(filename):
    """Read DEM file

    The DEM is provided as 16-bit signed integer data in a simple binary raster.
    There are no header or trailer bytes imbedded in the image.  The data are
    stored in row major order (all the data for row 1, followed by all the data
    for row 2, etc.).

    """
    with open(filename, "r") as f:
        a = np.fromfile(f, dtype="uint8")
    return a


def to_dataarray(data, nx, ny, types):
    """Create a data array from DEM data.

    Reshape data and flip into human readable format.
    Also add coordinate information.

    """
    x = np.linspace(-20015000.0, 20015000.0, nx)
    y = np.linspace(-8673000.0, 8673000.0, ny)
    data = np.flipud(data.reshape((ny, nx)))
    da = xr.DataArray(
        data=data,
        dims=("y", "x", "index"),
        coords=dict(x=(("x"), x), y=(("y"), y), index=(("index"), list(types.keys()))),
    )
    da.name = "type"
    return da.chunk({"x": 1000, "y": 1000})


def to_dataset(data, nx, ny, types):
    x = np.linspace(-20015000.0, 20015000.0, nx)
    y = np.linspace(-8673000.0, 8673000.0, ny)
    data = np.flipud(data.reshape((ny, nx)))  # .astype(np.float64)
    # data[data == 0.] = np.nan
    ds = xr.Dataset(
        data_vars=dict(
            glcc=(["y", "x"], data),
        ),
        coords=dict(
            x=(["x"], x),
            y=(["y"], y),
            index=(["index"], list(types.keys())),
            type=(["index"], list(types.values())),
            #      time=time,
            #      reference_time=reference_time,
        ),
        attrs=dict(
            description="Global Land Cover Characteristics Data Base Version 2.0."
        ),
    )
    ds.x.attrs["axis"] = "X"
    ds.y.attrs["axis"] = "Y"
    return ds.chunk({"x": 1000, "y": 1000})


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
