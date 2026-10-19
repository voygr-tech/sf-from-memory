"""The pictures in the README."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sfmap import geo, report, score


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


# water / park / building pass a colour-blind check against each other; street and other are dark and light greys
COLOURS = {"water": _rgb("#2a78d6"), "park": _rgb("#3d9b3d"), "building": _rgb("#e9a06b"),
           "street": _rgb("#3c3c3c"), "other": _rgb("#e4e1d8"), None: _rgb("#d6249f")}
PROVIDER = {"anthropic": "#2a78d6", "openai": "#eb6834", "gemini": "#1baf7a", None: "#8a8985"}
INK, MUTED = "#0b0b0b", "#52514e"


def _font(size):
    return ImageFont.load_default(size=size)


def map_image(grid, scale):
    rgb = np.array([[COLOURS.get(v, COLOURS[None]) for v in row] for row in grid], dtype=np.uint8)
    return Image.fromarray(rgb).resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)


def sheet(panels, title, path, cols, scale, three):
    """Map panels (title, grid) in a grid with a title and a legend."""
    rows_, cols_ = panels[0][1].shape
    w, h, pad, top, head, foot = cols_ * scale, rows_ * scale, 14, 24, 40, 34
    rows = -(-len(panels) // cols)
    img = Image.new("RGB", (cols * (w + pad) + pad, head + rows * (h + top + pad) + foot), "white")
    d = ImageDraw.Draw(img)
    d.text((pad, 12), title, fill=INK, font=_font(18))
    for i, (name, grid) in enumerate(panels):
        x0, y0 = pad + (i % cols) * (w + pad), head + (i // cols) * (h + top + pad)
        d.text((x0, y0 + 2), name, fill=INK, font=_font(14))
        img.paste(map_image(grid, scale), (x0, y0 + top - 4))
    legend = [("water", "water"), ("park", "park"), ("other", "everything else")] if three else \
        [("water", "water"), ("park", "park"), ("building", "building"), ("street", "street"), ("other", "other")]
    x, y = pad, img.height - foot + 8
    for cls, label in legend + [(None, "no answer")]:
        d.rectangle([x, y, x + 14, y + 14], fill=COLOURS[cls], outline=(150, 150, 150))
        d.text((x + 20, y), label, fill=MUTED, font=_font(14))
        x += 40 + int(d.textlength(label, font=_font(14)))
    img.save(path)
    return path


def cost_vs_score(rows, path):
    """rows: (model, level, provider, cost $, share right). One line per model through low, medium and high;
    point-at-a-time references as grey dots. Log cost axis."""
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    marker = {"low": "o", "medium": "s", "high": "^"}
    seen = {}
    lines = {}
    for model, level, provider, cost, right in rows:
        if level:
            lines.setdefault((model, provider), []).append((["low", "medium", "high"].index(level), cost, right * 100, level))
    for (model, provider), pts in lines.items():
        pts.sort()
        style = "-" if provider not in seen else "--"
        seen[provider] = True
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        ax.plot(xs, ys, style, color=PROVIDER[provider], linewidth=2, zorder=2)
        for _, x, y, level in pts:
            ax.scatter(x, y, s=55, marker=marker[level], color=PROVIDER[provider], edgecolor="white", linewidth=1.5,
                       zorder=3)
        nudge = {"Fable 5.1": -9, "Sol 6.1": -6, "Argon": 3}.get(model, -3)  # keep neighbouring end labels apart
        ax.annotate(model, (xs[-1], ys[-1]), xytext=(7, nudge), textcoords="offset points", fontsize=8.5, color=INK)
    low = []
    for model, level, provider, cost, right in rows:
        if level is None and right >= 0.6:
            ax.scatter(cost, right * 100, s=55, color=PROVIDER[None], edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(model, (cost, right * 100), xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=INK)
        elif level is None:
            low.append(f"{model}: {right:.0%} for ${cost:.2f}")
    if low:
        ax.text(0.01, 0.02, "Below the chart: " + "; ".join(low), transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_ylim(60, 90)
    ax.set_xlabel("Cost of the whole city map, $ (log scale)", color=MUTED)
    ax.set_ylabel("% of points right (water / park / everything else)", color=MUTED)
    for level, m in marker.items():
        ax.scatter([], [], s=55, marker=m, color=MUTED, label=f"{level} reasoning")
    for name, provider in (("Anthropic", "anthropic"), ("OpenAI", "openai"), ("Google", "gemini")):
        ax.plot([], [], color=PROVIDER[provider], linewidth=2, label=name)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5, ncol=2)
    ax.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("San Francisco at 100 m: what each model knows, and what it cost", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def medium_vs_high(rows, path):
    """rows: (model, street score at medium, at high or None). A dumbbell per model."""
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    for i, (model, med, high) in enumerate(rows):
        y = len(rows) - 1 - i
        if high is not None:
            ax.plot([med, high], [y, y], color="#c3c2b7", linewidth=2, zorder=1)
            ax.scatter(high, y, s=70, color=PROVIDER["openai"], edgecolor="white", linewidth=2, zorder=3)
        else:
            ax.annotate("high: ran out of room", (med, y), xytext=(10, -3), textcoords="offset points",
                        fontsize=8, color=MUTED)
        ax.scatter(med, y, s=70, color=PROVIDER["anthropic"], edgecolor="white", linewidth=2, zorder=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=9)
    ax.set_xlim(0, 0.4)
    ax.set_xlabel("Street score (IoU) on the 500 x 500 m block, 0 = none right, 1 = perfect", color=MUTED)
    ax.scatter([], [], s=70, color=PROVIDER["anthropic"], label="medium")
    ax.scatter([], [], s=70, color=PROVIDER["openai"], label="high")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", color="#e8e7e3", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("More reasoning barely moves the street score", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def all_figures(data_dir, out_dir, summary_path="results/summary.json"):
    out_dir.mkdir(exist_ok=True)
    summary = json.loads(Path(summary_path).read_text())
    paths = []

    def label(name, r, scheme):
        who = r["model"] + (f" · {r['level']}" if r["level"] else "")
        return f"{who} · {r[scheme]['accuracy']:.0%}"

    # all of San Francisco, water / park / everything else
    real, _ = report.load_reference(data_dir, "sf100")
    runs = report.runs(data_dir, "sf100")
    s = summary["maps"]["sf100"]["runs"]
    three = {k: score.fold3(v[2]) for k, v in runs.items()}
    hero = [("OpenStreetMap", score.fold3(real)), (label("astra-high", s["astra-high"], "three"), three["astra-high"]),
            (label("argon-high", s["argon-high"], "three"), three["argon-high"]), (label("jev", s["jev"], "three"), three["jev"])]
    paths.append(sheet(hero, "San Francisco from memory: water, parks and everything else, a point every 100 m",
                       out_dir / "hero.png", cols=4, scale=2, three=True))
    order = ["fable", "opus", "astra", "sol", "argon", "flash"]
    panels = [("OpenStreetMap", score.fold3(real)), (label("jev", s["jev"], "three"), three["jev"]),
              (label("decisions", s["decisions"], "three"), three["decisions"])]
    panels += [(label(f"{k}-{lvl}", s[f"{k}-{lvl}"], "three"), three[f"{k}-{lvl}"])
               for k in order for lvl in ("low", "medium", "high")]
    paths.append(sheet(panels, geo.MAPS["sf100"].title + " · water / park / everything else", out_dir / "sf.png",
                       cols=3, scale=2, three=True))

    # Crissy Field + Palace of Fine Arts + Marina, five classes
    real, _ = report.load_reference(data_dir, "marina10")
    runs = report.runs(data_dir, "marina10")
    s = summary["maps"]["marina10"]["runs"]
    panels = [("OpenStreetMap", real)] + [(label(k, s[k], "five"), runs[k][2])
                                          for k in [f"{m}-medium" for m in order] + ["jev", "decisions"]]
    paths.append(sheet(panels, geo.MAPS["marina10"].title + " · medium reasoning", out_dir / "marina.png",
                       cols=3, scale=2, three=False))

    # the 500 x 500 m block, medium next to high
    real, _ = report.load_reference(data_dir, "block10")
    runs = report.runs(data_dir, "block10")
    s = summary["maps"]["block10"]["runs"]
    panels = [("OpenStreetMap", real), (label("jev", s["jev"], "five"), runs["jev"][2]),
              (label("decisions", s["decisions"], "five"), runs["decisions"][2]), ("", np.full_like(real, "other"))]
    for m in order:
        panels += [(label(f"{m}-{lvl}", s[f"{m}-{lvl}"], "five"), runs[f"{m}-{lvl}"][2]) for lvl in ("medium", "high")]
    paths.append(sheet(panels[:3] + panels[4:], geo.MAPS["block10"].title + " · medium next to high",
                       out_dir / "block.png", cols=4, scale=4, three=False))

    # cost against score on the city, and medium against high on the block
    sf = summary["maps"]["sf100"]["runs"]
    from sfmap import models
    short = {"Claude Fable 5.1": "Fable 5.1", "Claude Opus 5.5": "Opus 5.5", "GPT-6 Astra": "Astra",
             "GPT-6.1 Sol": "Sol 6.1", "Gemini 4 Argon": "Argon", "Gemini 3.8 Flash": "Flash 3.8"}
    rows = [(short.get(r["model"], r["model"]), r["level"],
             models.MODELS[k.split("-")[0]].provider if r["level"] else None, r["cost_usd"], r["three"]["accuracy"])
            for k, r in sf.items()]
    paths.append(cost_vs_score(rows, out_dir / "cost_vs_score.png"))
    b = summary["maps"]["block10"]["runs"]
    rows = []
    for m in order:
        med, high = b[f"{m}-medium"], b[f"{m}-high"]
        rows.append((med["model"], med["five"]["iou"]["street"],
                     None if high["five"]["answered"] < 0.5 else high["five"]["iou"]["street"]))
    paths.append(medium_vs_high(rows, out_dir / "streets_medium_vs_high.png"))
    return paths
