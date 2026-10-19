"""Everything the README says, computed from the data folder: one summary.json."""

import json

import numpy as np

from sfmap import geo, models, prompt, score


def save_reference(data_dir, map_key, grid, retrieved):
    path = data_dir / "reference" / f"{map_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"map": map_key, "osm_retrieved": retrieved, "classes": grid.tolist()}))


def load_reference(data_dir, map_key):
    d = json.loads((data_dir / "reference" / f"{map_key}.json").read_text())
    return np.array(d["classes"], dtype=object), d["osm_retrieved"]


def runs(data_dir, map_key):
    """{run name: (label, level or None, grid, stats, cost $)} for every run of a map with answers on disk."""
    folder = data_dir / "answers" / map_key
    m = geo.MAPS[map_key]
    out = {}
    for kind, (label, price) in models.REFERENCES.items():
        recs = score.load(folder / f"{kind}.jsonl")
        if recs:
            out[kind] = (label, None, score.reference_grid(m, recs), {"requests": len(recs), "empty_first_try": 0,
                         "empty_final": 0, "tries": len(recs)}, sum(r["tin"] for r in recs) * price / 1e6)
    for key, model in models.MODELS.items():
        for level in m.levels:
            name = f"{key}-{level}"
            recs = score.load(folder / f"{name}.jsonl")
            if not recs:
                continue
            retries = score.load(folder / f"{name}.retries.jsonl")
            grid, stats = score.answer_grid(m, recs, retries)
            out[name] = (model.name, level, grid, stats, score.cost(model, recs, retries))
    return out


def _scores(pred, real, m, bootstrap_n):
    s = score.metrics(pred, real)
    s["range_95"] = score.bootstrap(pred, real, m, n=bootstrap_n)
    return s


def summarize(data_dir, bootstrap_n=1000):
    summary = {"maps": {}, "total_cost_usd": 0.0}
    for map_key, m in geo.MAPS.items():
        if not (data_dir / "answers" / map_key).exists():
            continue
        real, retrieved = load_reference(data_dir, map_key)
        real3 = score.fold3(real)
        entry = {"title": m.title, "step_m": m.step, "points": int(real.size), "osm_retrieved": retrieved,
                 "reference_shares": {c: float((real == c).mean()) for c in prompt.CLASSES}, "runs": {}}
        for name, (label, level, grid, stats, cost) in runs(data_dir, map_key).items():
            r = {"model": label, "level": level, **stats, "cost_usd": round(cost, 4),
                 "five": _scores(grid, real, m, bootstrap_n), "three": _scores(score.fold3(grid), real3, m, bootstrap_n),
                 "street_iou_shuffled": score.shuffled_iou(grid, real, "street", n=100),
                 "says_street": float(np.mean(grid[np.not_equal(grid, None)] == "street")) if np.not_equal(grid, None).any() else None}
            key = name.split("-")[0]
            rerun = score.load(data_dir / "answers" / map_key / f"{name}.rerun.jsonl")
            if rerun and key in models.MODELS:
                again, _ = score.answer_grid(m, rerun)
                r["rerun_agreement"] = score.agreement(grid, again)
                r["rerun_agreement_three"] = score.agreement(score.fold3(grid), score.fold3(again))
                r["rerun_three_accuracy"] = score.metrics(score.fold3(again), real3)["accuracy"]
                r["rerun_cost_usd"] = round(score.cost(models.MODELS[key], rerun), 4)
            elif rerun:
                again = score.reference_grid(m, rerun)
                r["rerun_agreement"] = score.agreement(grid, again)
                r["rerun_three_accuracy"] = score.metrics(score.fold3(again), real3)["accuracy"]
                r["top_probability"] = score.top_probability(m, rerun, real)
                r["rerun_cost_usd"] = round(sum(x["tin"] for x in rerun) * models.REFERENCES[key][1] / 1e6, 4)
            summary["total_cost_usd"] += r["cost_usd"] + r.get("rerun_cost_usd", 0)
            entry["runs"][name] = r
        summary["maps"][map_key] = entry
    summary["total_cost_usd"] = round(summary["total_cost_usd"], 4)
    return summary
