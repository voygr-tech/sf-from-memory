"""The answer key: what OpenStreetMap says is at each point.

A point is water if it is in the sea, a lake or a pond; else a building if it is inside a building footprint;
else a street if it is within half a road width of a street centreline; else a park if it is in a park, lawn,
wood or similar; else other (yards, parking lots, plazas, beaches, empty lots).
"""

import itertools
import json
import math
import time

import numpy as np
import requests
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union

from sfmap import geo

OVERPASS = ["https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
QUERY = """[out:json][timeout:300];
(
  way["highway"]({b}); way["building"]({b}); relation["building"]({b});
  way["natural"="coastline"]({b});
  way["natural"~"^(water|wood|scrub|grassland|heath)$"]({b}); relation["natural"~"^(water|wood)$"]({b});
  way["water"]({b}); way["landuse"~"^(basin|reservoir|grass|recreation_ground|meadow|village_green|forest)$"]({b});
  relation["landuse"~"^(grass|recreation_ground|forest)$"]({b});
  way["leisure"~"^(park|garden|playground|pitch|recreation_ground|golf_course|nature_reserve|common|dog_park|marina)$"]({b});
  relation["leisure"~"^(park|garden|nature_reserve|golf_course|recreation_ground|marina)$"]({b});
);
out geom;"""

HALF_WIDTH_M = {"primary": 12, "secondary": 12, "tertiary": 12, "residential": 10, "unclassified": 10,
                "living_street": 6, "pedestrian": 6, "service": 5}  # metres either side of the centreline
WATER_TAGS = {("natural", "water"), ("landuse", "basin"), ("landuse", "reservoir"), ("leisure", "marina")}
PARK = {"leisure": {"park", "garden", "playground", "pitch", "recreation_ground", "golf_course", "nature_reserve",
                    "common", "dog_park"},
        "landuse": {"grass", "recreation_ground", "meadow", "village_green", "forest"},
        "natural": {"wood", "scrub", "grassland", "heath"}}
FRAME_HALF = 7000  # metres: the square around the city that the download and the coastline use
MARGIN = 150  # metres beyond the frame, so shapes crossing its edge are complete


def tiles(bbox, n):
    """bbox (south, west, north, east) cut into n x n tiles."""
    s, w, north, e = bbox
    return [(s + (north - s) * i / n, w + (e - w) * j / n, s + (north - s) * (i + 1) / n, w + (e - w) * (j + 1) / n)
            for i in range(n) for j in range(n)]


def fetch(path, n=4):
    """Download the city from Overpass in n x n tiles (one request is too big) and save the merged elements."""
    s, w = geo.unproject(-FRAME_HALF - MARGIN, -FRAME_HALF - MARGIN)
    north, e = geo.unproject(FRAME_HALF + MARGIN, FRAME_HALF + MARGIN)
    seen, elements = set(), []
    for t in tiles((s, w, north, e), n):
        q = QUERY.format(b=",".join(f"{v:.6f}" for v in t))
        for attempt in range(6):
            try:
                r = requests.post(OVERPASS[attempt % len(OVERPASS)], data={"data": q}, timeout=400,
                                  headers={"User-Agent": "sf-from-memory/0.1"})
                if r.status_code == 200 and r.content.lstrip().startswith(b"{"):
                    break
            except requests.RequestException:
                pass
            time.sleep(20)
        else:
            raise RuntimeError(f"no Overpass server answered for tile {t}")
        for el in r.json()["elements"]:
            if (el["type"], el["id"]) not in seen:
                seen.add((el["type"], el["id"]))
                elements.append(el)
    path.write_text(json.dumps({"elements": elements}))
    return len(elements)


def _line(geometry):
    return [geo.project(p["lat"], p["lon"]) for p in geometry if p]


def _area(el):
    if el["type"] == "way":
        pts = _line(el["geometry"])
        return shapely.make_valid(Polygon(pts)) if len(pts) >= 4 and pts[0] == pts[-1] else None
    outer = [LineString(_line(m["geometry"])) for m in el.get("members", [])
             if m.get("role") == "outer" and len(m.get("geometry") or []) >= 2]
    inner = [LineString(_line(m["geometry"])) for m in el.get("members", [])
             if m.get("role") == "inner" and len(m.get("geometry") or []) >= 2]
    try:  # a few relations have broken rings; repair them or skip them
        shape = unary_union([shapely.make_valid(p) for p in polygonize(outer)])
        return None if shape.is_empty else shape.difference(unary_union([shapely.make_valid(p) for p in polygonize(inner)]))
    except shapely.errors.GEOSException:
        return None


def sea_by_vote(coast, half=FRAME_HALF):
    """The sea, from coastline ways. OpenStreetMap draws coastlines with land on the left and water on the right.
    The frame is cut along the coastlines; each piece is sea when more probes from the water side than from the
    land side land inside it (a vote, because single probes misfire on piers and sharp corners)."""
    if not coast:
        return shapely.GeometryCollection()
    frame = box(-half - MARGIN, -half - MARGIN, half + MARGIN, half + MARGIN)
    noded = unary_union([frame.exterior] + [g for g in (c.intersection(frame) for c in coast) if not g.is_empty])
    right, left = [], []
    for c in coast:
        pts = list(c.coords)
        for (x0, y0), (x1, y1) in itertools.pairwise(pts):
            d = math.hypot(x1 - x0, y1 - y0)
            if d > 0:
                mx, my, nx, ny = (x0 + x1) / 2, (y0 + y1) / 2, (y1 - y0) / d, -(x1 - x0) / d
                right.append(Point(mx + 3 * nx, my + 3 * ny))
                left.append(Point(mx - 3 * nx, my - 3 * ny))
    sea = [poly for poly in polygonize(noded)
           if sum(poly.contains(p) for p in right) > sum(poly.contains(p) for p in left)]
    return unary_union(sea) if sea else shapely.GeometryCollection()


def shapes(elements, half=FRAME_HALF):
    """water, park, building and street areas (shapely geometries, in metres) from Overpass elements."""
    water, park, building, streets, coast = [], [], [], [], []
    for el in elements:
        tags = el.get("tags", {})
        if el["type"] == "way" and tags.get("natural") == "coastline":
            coast.append(LineString(_line(el["geometry"])))
            continue
        if (el["type"] == "way" and tags.get("highway") in HALF_WIDTH_M and "area" not in tags
                and (tags["highway"] != "service" or tags.get("service") in (None, "alley"))):
            streets.append(LineString(_line(el["geometry"])).buffer(HALF_WIDTH_M[tags["highway"]]))
            continue
        if not any(k in tags for k in ("building", "natural", "water", "landuse", "leisure")):
            continue
        shape = _area(el)
        if shape is None:
            continue
        if tags.get("building", "no") != "no":
            building.append(shape)
        elif any((k, tags.get(k)) in WATER_TAGS for k in ("natural", "landuse", "leisure")) or "water" in tags:
            water.append(shape)
        elif any(tags.get(k) in v for k, v in PARK.items()):
            park.append(shape)
    inland = unary_union(water)
    small = [g for g in getattr(inland, "geoms", [inland]) if not g.is_empty and g.area < 5e6]  # lakes, ponds, harbours
    return {"water": unary_union([sea_by_vote(coast, half)] + small), "park": unary_union(park),
            "building": unary_union(building), "street": unary_union(streets)}


def classify(s, xs, ys):
    """Class at each (x, y). Later layers win: other < park < street < building < water."""
    out = np.full(xs.shape, "other", dtype=object)
    for cls in ("park", "street", "building", "water"):
        g = s[cls]
        if not g.is_empty:
            shapely.prepare(g)
            out[shapely.contains_xy(g, xs, ys)] = cls
    return out


def reference(m, s):
    """The answer key for a map: a grid of classes, rows north to south."""
    xs, ys = geo.axes(m)
    X, Y = np.meshgrid(xs, ys)
    return classify(s, X, Y)
