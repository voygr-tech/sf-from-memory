import numpy as np
import shapely

from sfmap import geo, osm


def way(tags, coords_m):
    """An Overpass way with geometry, from (x, y) metres around the city centre."""
    pts = [geo.unproject(x, y) for x, y in coords_m]
    return {"type": "way", "id": abs(hash(str(coords_m))) % 10**9, "tags": tags,
            "geometry": [{"lat": lat, "lon": lon} for lat, lon in pts]}


def square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def classes_at(shapes, xy):
    xs = np.array([[x for x, _ in xy]], dtype=float)
    ys = np.array([[y for _, y in xy]], dtype=float)
    return list(osm.classify(shapes, xs, ys)[0])


def test_shapes_sorts_features_into_classes():
    elements = [way({"building": "yes"}, square(0, 0, 10, 10)),
                way({"leisure": "park"}, square(100, 0, 200, 100)),
                way({"natural": "water"}, square(300, 0, 400, 100)),
                way({"highway": "residential", "name": "Chestnut Street"}, [(-100, 50), (-100, 150)])]
    s = osm.shapes(elements, half=1000)
    assert classes_at(s, [(5, 5), (150, 50), (350, 50), (-100, 100), (-500, -500)]) == \
        ["building", "park", "water", "street", "other"]


def test_streets_are_buffered_by_their_type():
    elements = [way({"highway": "primary"}, [(0, -100), (0, 100)]),
                way({"highway": "residential"}, [(500, -100), (500, 100)])]
    s = osm.shapes(elements, half=1000)
    # primary roads count as 12 m either side, residential streets as 10 m
    assert classes_at(s, [(11, 0), (13, 0), (509, 0), (511, 0)]) == ["street", "other", "street", "other"]


def test_driveways_and_footpaths_are_not_streets_but_alleys_are():
    elements = [way({"highway": "service", "service": "driveway"}, [(0, -50), (0, 50)]),
                way({"highway": "footway"}, [(100, -50), (100, 50)]),
                way({"highway": "service", "service": "alley"}, [(200, -50), (200, 50)])]
    s = osm.shapes(elements, half=1000)
    assert classes_at(s, [(0, 0), (100, 0), (200, 0)]) == ["other", "other", "street"]


def test_priority_water_over_building_over_street_over_park():
    elements = [way({"leisure": "park"}, square(0, 0, 100, 100)),
                way({"highway": "residential"}, [(50, -50), (50, 150)]),
                way({"building": "yes"}, square(45, 45, 55, 55)),
                way({"natural": "water"}, square(48, 48, 52, 52))]
    s = osm.shapes(elements, half=1000)
    assert classes_at(s, [(20, 20), (50, 20), (46, 50), (50, 50)]) == ["park", "street", "building", "water"]


def test_sea_by_vote_floods_the_water_side_of_a_coastline():
    # OpenStreetMap coastlines have land on the left and water on the right: walking east, the sea is south
    coast = [shapely.LineString([(-2000, 0), (2000, 0)])]
    sea = osm.sea_by_vote(coast, half=1000)
    assert shapely.contains_xy(sea, 0, -500) and not shapely.contains_xy(sea, 0, 500)


def test_sea_by_vote_without_coastline_is_empty():
    assert osm.sea_by_vote([], half=1000).is_empty


def test_shapes_turns_coastline_into_sea_and_skips_huge_inland_water():
    elements = [way({"natural": "coastline"}, [(-2000, 0), (2000, 0)])]
    s = osm.shapes(elements, half=1000)
    assert classes_at(s, [(0, -500), (0, 500)]) == ["water", "other"]


def test_broken_relation_is_skipped_not_fatal():
    broken = {"type": "relation", "id": 1, "tags": {"leisure": "park"},
              "members": [{"role": "outer", "geometry": [{"lat": 37.76, "lon": -122.43}]}]}
    s = osm.shapes([broken], half=1000)
    assert classes_at(s, [(0, 0)]) == ["other"]


def test_tiles_cover_the_box_without_gaps():
    tiles = osm.tiles((37.70, -122.52, 37.82, -122.35), 4)
    assert len(tiles) == 16
    assert min(t[0] for t in tiles) == 37.70 and max(t[2] for t in tiles) == 37.82
    assert min(t[1] for t in tiles) == -122.52 and max(t[3] for t in tiles) == -122.35
