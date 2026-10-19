import numpy as np
from PIL import Image

from sfmap import figures


def test_map_image_colours_each_class_and_no_answer():
    grid = np.array([["water", "park", "building"], ["street", "other", None]], dtype=object)
    img = figures.map_image(grid, scale=2)
    assert img.size == (6, 4)
    assert img.getpixel((0, 0)) == figures.COLOURS["water"]
    assert img.getpixel((5, 3)) == figures.COLOURS[None]


def test_sheet_lays_panels_out_in_columns(tmp_path):
    grid = np.full((10, 20), "park", dtype=object)
    path = tmp_path / "sheet.png"
    figures.sheet([("a", grid), ("b", grid), ("c", grid)], "Title", path, cols=2, scale=3, three=False)
    w, h = Image.open(path).size
    assert w >= 2 * 20 * 3 and h >= 2 * 10 * 3


def test_cost_vs_score_chart_writes_a_png(tmp_path):
    rows = [("GPT-6 Astra", "high", "openai", 135.0, 0.864), ("Jev", None, None, 0.3, 0.667)]
    path = tmp_path / "c.png"
    figures.cost_vs_score(rows, path)
    assert Image.open(path).size[0] > 400


def test_dumbbell_chart_writes_a_png(tmp_path):
    rows = [("GPT-6 Astra", 0.31, 0.32), ("Gemini 4 Argon", 0.22, None)]
    path = tmp_path / "d.png"
    figures.medium_vs_high(rows, path)
    assert Image.open(path).size[0] > 400


def test_colours_cover_every_class():
    assert set(figures.COLOURS) == {"water", "park", "building", "street", "other", None}
