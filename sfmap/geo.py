"""Where the points are: a flat local frame around San Francisco, the three maps, and the shuffled requests."""

import math
import random
from dataclasses import dataclass

import numpy as np

CENTER = (37.7595, -122.436)
HALF_X, HALF_Y = 6950, 5725  # metres: a 13.9 x 11.45 km rectangle that holds the whole city
M_LAT = 6_371_008.8 * math.pi / 180  # metres per degree of latitude
M_LON = M_LAT * math.cos(math.radians(CENTER[0]))


def project(lat, lon):
    """Degrees to metres east (x) and north (y) of CENTER."""
    return (lon - CENTER[1]) * M_LON, (lat - CENTER[0]) * M_LAT


def unproject(x, y):
    return CENTER[0] + y / M_LAT, CENTER[1] + x / M_LON


@dataclass(frozen=True)
class Map:
    key: str
    title: str
    step: int  # metres between points
    box: tuple | None  # (south, west, north, east), or None for the whole city
    levels: tuple  # reasoning levels each model runs at


MAPS = {
    "sf100": Map("sf100", "All of San Francisco, a point every 100 m", 100, None, ("low", "medium", "high")),
    "marina10": Map("marina10", "Crissy Field, the Palace of Fine Arts and the Marina, a point every 10 m", 10,
                    (37.7982, -122.4559, 37.8092, -122.4343), ("medium",)),
    "block10": Map("block10", "A 500 x 500 m piece of the Marina street grid, a point every 10 m", 10,
                   (37.799757, -122.443956, 37.804254, -122.438268), ("medium", "high")),
}


def axes(m):
    """x of each column (west to east) and y of each row (north to south), in metres."""
    if m.box is None:
        return (np.arange(-HALF_X + m.step / 2, HALF_X, m.step),
                np.arange(HALF_Y - m.step / 2, -HALF_Y, -m.step))
    south, west, north, east = m.box
    (x0, y0), (x1, y1) = project(north, west), project(south, east)
    return np.arange(x0 + m.step / 2, x1, m.step), np.arange(y0 - m.step / 2, y1, -m.step)


def points(m):
    """Every point of a map as {id, r, c, x, y, lat, lon}, row by row from the north-west corner."""
    xs, ys = axes(m)
    pts = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            lat, lon = unproject(x, y)
            pts.append({"id": f"{r:03d}:{c:03d}", "r": r, "c": c, "x": float(x), "y": float(y), "lat": lat, "lon": lon})
    return pts, len(ys), len(xs)


def requests(m, size=100, seed=390):
    """The points shuffled once, with a fixed seed, and cut into requests of `size` points."""
    if size < 1:
        raise ValueError("size must be at least 1")
    pts, _, _ = points(m)
    random.Random(seed).shuffle(pts)
    return [pts[i:i + size] for i in range(0, len(pts), size)]
