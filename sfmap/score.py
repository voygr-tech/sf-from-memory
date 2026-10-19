"""From raw answers to scores: answer grids, cost, accuracy, IoU, bootstrap ranges and run-to-run agreement."""

import json

import numpy as np

from sfmap import geo, prompt

CLASSES_3 = ["water", "park", "other"]


def load(path):
    """Records of a JSONL file; broken lines are skipped, a missing file is empty."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def answer_grid(m, records, retries=()):
    """Grid of classes (None = no answer) from one record per request. A request that came back incomplete
    is replaced by its first complete retry, if there is one."""
    _, rows, cols = geo.points(m)
    groups = geo.requests(m)
    grid = np.full((rows, cols), None, dtype=object)
    extra = {}
    for r in retries:
        extra.setdefault(r["k"], []).append(r)
    stats = {"requests": len(records), "empty_first_try": 0, "empty_final": 0, "tries": len(records) + len(retries)}
    for rec in records:
        g = groups[rec["k"]]
        got = prompt.parse(rec["text"], len(g))
        if len(got) < len(g):
            stats["empty_first_try"] += 1
            done = next((p for p in (prompt.parse(x["text"], len(g)) for x in extra.get(rec["k"], []))
                         if len(p) == len(g)), None)
            if done is None:
                stats["empty_final"] += 1
            else:
                got = done
        for i, cls in got.items():
            grid[g[i]["r"], g[i]["c"]] = cls
    return grid, stats


def reference_grid(m, records):
    """Grid from point-at-a-time answers ({id, choice}), as Jev and the Decisions API give them."""
    pts, rows, cols = geo.points(m)
    where = {p["id"]: (p["r"], p["c"]) for p in pts}
    grid = np.full((rows, cols), None, dtype=object)
    for rec in records:
        if rec.get("choice") in prompt.CLASSES:
            grid[where[rec["id"]]] = rec["choice"]
    return grid


def top_probability(m, records, real):
    """How sure a point-at-a-time model was of the answer it gave, on average: overall, when right and when wrong."""
    where = {p["id"]: (p["r"], p["c"]) for p in geo.points(m)[0]}
    right, wrong = [], []
    for rec in records:
        if rec.get("probs") and rec.get("choice") in rec["probs"]:
            (right if real[where[rec["id"]]] == rec["choice"] else wrong).append(rec["probs"][rec["choice"]])
    mean = lambda v: float(np.mean(v)) if v else None
    return {"mean": mean(right + wrong), "when_right": mean(right), "when_wrong": mean(wrong)}


def cost(model, records, retries=()):
    """Dollars for every try. Batch runs pay half, except requests that went through the regular API."""
    total = 0.0
    for r in list(records) + list(retries):
        half = model.batch and r.get("priced") != "standard"
        total += (r["tin"] * model.price_in + r["tout"] * model.price_out) / 1e6 * (0.5 if half else 1)
    return total


def fold3(a):
    """building, street and other become one class: water / park / other."""
    return np.vectorize(lambda v: v if v in (None, "water", "park") else "other", otypes=[object])(a)


def metrics(pred, real, classes=None):
    """Share right among answered points, share answered, IoU per class and mean IoU over the classes in the key."""
    classes = classes or sorted(set(prompt.CLASSES) | set(CLASSES_3))
    answered = np.not_equal(pred, None)
    acc = float((pred[answered] == real[answered]).mean()) if answered.any() else None
    iou = {}
    for c in classes:
        union = ((pred == c) | (real == c)).sum()
        iou[c] = float(((pred == c) & (real == c)).sum() / union) if union else None
    present = [v for c, v in iou.items() if v is not None and (real == c).any()]
    return {"accuracy": acc, "answered": float(answered.mean()), "iou": iou,
            "mean_iou": float(np.mean(present)) if present else None}


def bootstrap(pred, real, m, n=1000, seed=390):
    """95% ranges for accuracy and mean IoU, resampling whole requests (points in one request are not independent)."""
    names = sorted({v for v in real.ravel()} | {v for v in pred.ravel() if v is not None})
    code = {c: i for i, c in enumerate(names)}
    p = np.array([code.get(v, -1) for v in pred.ravel()])  # -1 = no answer
    r = np.array([code[v] for v in real.ravel()])
    cols = pred.shape[1]
    cells = [np.array([q["r"] * cols + q["c"] for q in g]) for g in geo.requests(m)]
    rng = np.random.default_rng(seed)
    accs, ious = [], []
    for _ in range(n):
        idx = np.concatenate([cells[i] for i in rng.integers(0, len(cells), len(cells))])
        pp, rr = p[idx], r[idx]
        answered = pp >= 0
        if not answered.any():
            continue
        accs.append(float((pp[answered] == rr[answered]).mean()))
        per = []
        for c in np.unique(rr):
            union = ((pp == c) | (rr == c)).sum()
            per.append(((pp == c) & (rr == c)).sum() / union)
        ious.append(float(np.mean(per)))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (None, None)
    return {"accuracy": q(accs), "mean_iou": q(ious)}


def agreement(a, b):
    """Share of points where two runs give the same class, among points both answered."""
    both = np.not_equal(a, None) & np.not_equal(b, None)
    return float((a[both] == b[both]).mean()) if both.any() else None


def shuffled_iou(pred, real, cls, n=200, seed=390):
    """IoU for one class if the model's own answers were shuffled across the points it answered: the score you
    get by saying that class equally often but in random places."""
    answered = np.not_equal(pred, None)
    p, r = pred[answered], real[answered]
    if not (p == cls).any():
        return 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        q = rng.permutation(p)
        union = ((q == cls) | (r == cls)).sum()
        vals.append(((q == cls) & (r == cls)).sum() / union if union else 0.0)
    return float(np.mean(vals))
