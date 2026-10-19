import json

import numpy as np
import pytest

from sfmap import geo, models, score

M = geo.MAPS["block10"]  # 2,500 points in 25 requests


def full_answer(group, letter="H"):
    return json.dumps({"codes": [f"{i + 1}:{letter}" for i in range(len(group))]})


def records(letter="H", skip=()):
    return [{"k": k, "text": "" if k in skip else full_answer(g, letter), "tin": 1000, "tout": 2000}
            for k, g in enumerate(geo.requests(M))]


def test_load_skips_broken_lines(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"k": 0}\nnot json\n{"k": 1}\n')
    assert score.load(p) == [{"k": 0}, {"k": 1}]


def test_load_missing_file_is_empty(tmp_path):
    assert score.load(tmp_path / "missing.jsonl") == []


def test_answer_grid_places_every_answer_on_its_point():
    grid, stats = score.answer_grid(M, records("S"))
    assert grid.shape == (50, 50) and (grid == "street").all()
    assert stats == {"requests": 25, "empty_first_try": 0, "empty_final": 0, "tries": 25}


def test_answer_grid_leaves_empty_requests_as_no_answer():
    grid, stats = score.answer_grid(M, records(skip={3, 7}))
    assert np.equal(grid, None).sum() == 200
    assert stats["empty_first_try"] == 2 and stats["empty_final"] == 2


def test_answer_grid_uses_first_complete_retry():
    groups = geo.requests(M)
    retries = [{"k": 3, "attempt": 2, "text": ""}, {"k": 3, "attempt": 3, "text": full_answer(groups[3], "W")}]
    grid, stats = score.answer_grid(M, records(skip={3}), retries)
    assert np.equal(grid, None).sum() == 0
    assert (grid == "water").sum() == len(groups[3])
    assert stats == {"requests": 25, "empty_first_try": 1, "empty_final": 0, "tries": 27}


def test_reference_grid_from_point_answers():
    pts, _, _ = geo.points(M)
    recs = [{"id": p["id"], "choice": "park"} for p in pts[:10]]
    grid = score.reference_grid(M, recs)
    assert (grid == "park").sum() == 10 and np.equal(grid, None).sum() == len(pts) - 10


def test_cost_counts_retries_and_standard_priced_fills():
    m = models.MODELS["sol"]  # $2 / $10 per 1M, batch at half
    recs = [{"tin": 1_000_000, "tout": 0}, {"tin": 1_000_000, "tout": 0, "priced": "standard"}]
    assert score.cost(m, recs) == pytest.approx(1 + 2)
    gem = models.MODELS["argon"]  # never batch
    assert score.cost(gem, [{"tin": 0, "tout": 1_000_000}], [{"tin": 0, "tout": 1_000_000}]) == pytest.approx(20)


def test_metrics_perfect_and_partial():
    real = np.array([["water", "park"], ["street", "other"]], dtype=object)
    assert score.metrics(real.copy(), real)["accuracy"] == 1.0
    pred = np.array([["water", None], ["park", "other"]], dtype=object)
    m = score.metrics(pred, real)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["answered"] == pytest.approx(3 / 4)
    assert m["iou"]["water"] == 1.0 and m["iou"]["street"] == 0.0
    assert m["iou"]["building"] is None  # not in the answer key and never said


def test_metrics_mean_iou_only_over_classes_in_the_answer_key():
    real = np.array([["water", "water"]], dtype=object)
    pred = np.array([["water", "park"]], dtype=object)
    m = score.metrics(pred, real)
    assert m["mean_iou"] == pytest.approx(0.5)  # water only; park is not in the key


def test_metrics_nothing_answered():
    real = np.array([["water"]], dtype=object)
    m = score.metrics(np.array([[None]], dtype=object), real)
    assert m["accuracy"] is None and m["answered"] == 0


def test_fold3_merges_building_street_and_other():
    a = np.array([["water", "park", "building", "street", "other", None]], dtype=object)
    assert list(score.fold3(a)[0]) == ["water", "park", "other", "other", "other", None]


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    real = score.fold3(np.full((50, 50), "building", dtype=object))
    pred = real.copy()
    pred[:10] = "park"
    lo, hi = score.bootstrap(pred, real, M, n=200)["accuracy"]
    acc = score.metrics(pred, real)["accuracy"]
    assert lo <= acc <= hi
    assert score.bootstrap(pred, real, M, n=200) == score.bootstrap(pred, real, M, n=200)


def test_agreement_counts_points_both_runs_answered():
    a = np.array([["water", "park", None]], dtype=object)
    b = np.array([["water", "other", "park"]], dtype=object)
    assert score.agreement(a, b) == pytest.approx(0.5)


def test_shuffled_iou_is_the_chance_level_for_a_class():
    real = np.array([["street"] * 50 + ["building"] * 50], dtype=object)
    pred = np.array([["street"] * 50 + ["building"] * 50], dtype=object)  # perfect, street IoU 1
    chance = score.shuffled_iou(pred, real, "street", n=200)
    # shuffling 50 street answers over 100 points: expected overlap 25, union 75
    assert chance == pytest.approx(1 / 3, abs=0.03)


def test_shuffled_iou_ignores_unanswered_points_and_is_reproducible():
    real = np.array([["street", "street", "building", "building"]], dtype=object)
    pred = np.array([["street", None, "building", None]], dtype=object)
    a = score.shuffled_iou(pred, real, "street", n=50)
    assert a == score.shuffled_iou(pred, real, "street", n=50)
    assert 0 <= a <= 1


def test_shuffled_iou_never_said_class_is_zero():
    real = np.array([["street", "building"]], dtype=object)
    pred = np.array([["building", "building"]], dtype=object)
    assert score.shuffled_iou(pred, real, "street", n=10) == 0


def test_top_probability_splits_right_and_wrong_answers():
    pts, rows, cols = geo.points(M)
    real = np.full((rows, cols), "street", dtype=object)
    recs = [{"id": pts[0]["id"], "choice": "street", "probs": {"street": 0.9, "water": 0.1}},
            {"id": pts[1]["id"], "choice": "water", "probs": {"street": 0.3, "water": 0.7}},
            {"id": pts[2]["id"], "choice": "water", "probs": {"street": 0.5, "water": 0.5}}]
    t = score.top_probability(M, recs, real)
    assert t["mean"] == pytest.approx(0.7)
    assert t["when_right"] == pytest.approx(0.9) and t["when_wrong"] == pytest.approx(0.6)


def test_top_probability_without_probabilities_is_empty():
    pts, rows, cols = geo.points(M)
    real = np.full((rows, cols), "street", dtype=object)
    t = score.top_probability(M, [{"id": pts[0]["id"], "choice": "street", "probs": None}], real)
    assert t == {"mean": None, "when_right": None, "when_wrong": None}
