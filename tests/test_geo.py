import pytest

from sfmap import geo


def test_project_unproject_round_trip():
    lat, lon = 37.8024, -122.4485  # Palace of Fine Arts
    x, y = geo.project(lat, lon)
    assert geo.unproject(x, y) == pytest.approx((lat, lon), abs=1e-9)


def test_project_center_is_origin():
    assert geo.project(*geo.CENTER) == (0.0, 0.0)


def test_project_one_km_east_and_north():
    lat, lon = geo.unproject(1000, 1000)
    x, y = geo.project(lat, lon)
    assert (x, y) == pytest.approx((1000, 1000))
    assert lat > geo.CENTER[0] and lon > geo.CENTER[1]


def test_points_city_map_has_expected_grid():
    pts, rows, cols = geo.points(geo.MAPS["sf100"])
    assert (rows, cols) == (114, 139)
    assert len(pts) == rows * cols == 15_846


def test_points_rows_go_north_to_south_and_columns_west_to_east():
    pts, _, _ = geo.points(geo.MAPS["block10"])
    first, last = pts[0], pts[-1]
    assert (first["r"], first["c"]) == (0, 0)
    assert first["lat"] > last["lat"] and first["lon"] < last["lon"]


def test_points_box_maps_stay_inside_their_box():
    m = geo.MAPS["marina10"]
    south, west, north, east = m.box
    pts, _, _ = geo.points(m)
    assert all(south < p["lat"] < north and west < p["lon"] < east for p in pts)


def test_points_ids_are_unique_row_column_codes():
    pts, _, _ = geo.points(geo.MAPS["block10"])
    assert len({p["id"] for p in pts}) == len(pts)
    assert pts[0]["id"] == "000:000"


def test_requests_cover_every_point_once_in_groups_of_100():
    m = geo.MAPS["marina10"]
    pts, _, _ = geo.points(m)
    groups = geo.requests(m)
    assert [len(g) for g in groups[:-1]] == [100] * (len(groups) - 1)
    assert 0 < len(groups[-1]) <= 100
    assert sorted(p["id"] for g in groups for p in g) == sorted(p["id"] for p in pts)


def test_requests_are_shuffled_and_reproducible():
    m = geo.MAPS["block10"]
    a, b = geo.requests(m), geo.requests(m)
    assert [p["id"] for p in a[0]] == [p["id"] for p in b[0]]
    pts, _, _ = geo.points(m)
    assert [p["id"] for p in a[0]] != [p["id"] for p in pts[:100]]


def test_requests_rejects_non_positive_size():
    with pytest.raises(ValueError):
        geo.requests(geo.MAPS["block10"], size=0)
