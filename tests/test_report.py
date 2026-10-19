import json

import numpy as np
import pytest

from sfmap import geo, report

M = geo.MAPS["block10"]


def write_jsonl(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


@pytest.fixture
def data(tmp_path):
    """A tiny data folder: the block map, all street; Sol medium answers street everywhere (twice), Jev says water."""
    rows, cols = 50, 50
    report.save_reference(tmp_path, "block10", np.full((rows, cols), "street", dtype=object), "2026-10-18")
    groups = geo.requests(M)
    sol = [{"k": k, "text": json.dumps({"codes": [f"{i + 1}:S" for i in range(len(g))]}), "tin": 1000, "tout": 1000}
           for k, g in enumerate(groups)]
    write_jsonl(tmp_path / "answers/block10/sol-medium.jsonl", sol)
    write_jsonl(tmp_path / "answers/block10/sol-medium.rerun.jsonl", sol)
    pts, _, _ = geo.points(M)
    write_jsonl(tmp_path / "answers/block10/jev.jsonl", [{"id": p["id"], "choice": "water", "tin": 400} for p in pts])
    write_jsonl(tmp_path / "answers/block10/jev.rerun.jsonl",
                [{"id": p["id"], "choice": "water", "tin": 400, "probs": {"water": 0.4, "street": 0.35}} for p in pts])
    return tmp_path


def test_save_and_load_reference_round_trip(tmp_path):
    grid = np.array([["water", "park"], ["street", "other"]], dtype=object)
    report.save_reference(tmp_path, "block10", grid, "2026-10-18")
    loaded, retrieved = report.load_reference(tmp_path, "block10")
    assert (loaded == grid).all() and retrieved == "2026-10-18"


def test_summarize_scores_each_run(data):
    s = report.summarize(data, bootstrap_n=50)
    sol = s["maps"]["block10"]["runs"]["sol-medium"]
    assert sol["model"] == "GPT-6.1 Sol" and sol["level"] == "medium"
    assert sol["five"]["accuracy"] == 1.0 and sol["three"]["accuracy"] == 1.0
    assert sol["requests"] == 25 and sol["empty_final"] == 0
    assert sol["cost_usd"] == pytest.approx(25 * (1000 * 2 + 1000 * 10) / 1e6 * 0.5)


def test_summarize_scores_references(data):
    jev = report.summarize(data, bootstrap_n=50)["maps"]["block10"]["runs"]["jev"]
    assert jev["model"] == "Jev" and jev["five"]["accuracy"] == 0.0


def test_summarize_measures_run_to_run_agreement(data):
    s = report.summarize(data, bootstrap_n=50)
    assert s["maps"]["block10"]["runs"]["sol-medium"]["rerun_agreement"] == 1.0


def test_summarize_totals_cost(data):
    s = report.summarize(data, bootstrap_n=50)
    runs = s["maps"]["block10"]["runs"]
    spent = sum(r["cost_usd"] + r.get("rerun_cost_usd", 0) for r in runs.values())
    assert s["total_cost_usd"] == pytest.approx(spent, abs=1e-3)
    assert runs["sol-medium"]["rerun_cost_usd"] == pytest.approx(runs["sol-medium"]["cost_usd"])


def test_summarize_skips_maps_without_answers(data):
    assert set(report.summarize(data, bootstrap_n=50)["maps"]) == {"block10"}


def test_summarize_reports_street_chance_level(data):
    sol = report.summarize(data, bootstrap_n=50)["maps"]["block10"]["runs"]["sol-medium"]
    assert sol["says_street"] == 1.0
    assert sol["street_iou_shuffled"] == pytest.approx(1.0)  # all street in an all-street key: shuffling changes nothing


def test_summarize_reports_reference_rerun_and_how_sure_it_was(data):
    jev = report.summarize(data, bootstrap_n=50)["maps"]["block10"]["runs"]["jev"]
    assert jev["rerun_agreement"] == 1.0
    assert jev["top_probability"]["when_wrong"] == pytest.approx(0.4) and jev["top_probability"]["when_right"] is None
    assert jev["rerun_cost_usd"] == pytest.approx(jev["cost_usd"])
    assert jev["rerun_three_accuracy"] == 0.0  # water everywhere on an all-street map
