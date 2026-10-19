"""python -m sfmap <command>. See the README, "Run it yourself"."""

import argparse
import asyncio
import datetime
import io
import json
import zipfile
from pathlib import Path

import requests

from sfmap import geo, report, run

DATA_URL = "https://github.com/voygr-tech/sf-from-memory/releases/download/v1/data.zip"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sfmap")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="the answers and answer keys behind the report (data/)")
    sub.add_parser("reference", help="download OpenStreetMap for the city and build the answer keys")
    r = sub.add_parser("run", help="ask every model about a map (Batch APIs, Gemini, Jev, Decisions API)")
    r.add_argument("map", choices=geo.MAPS)
    f = sub.add_parser("references", help="ask Jev and the Decisions API only, keeping their probabilities")
    f.add_argument("map", choices=geo.MAPS)
    f.add_argument("--rerun", action="store_true", help="a whole second run, saved as <name>.rerun.jsonl")
    sub.add_parser("collect", help="download finished batches")
    t = sub.add_parser("retry", help="ask again, up to 4 more times, Gemini requests that came back empty")
    b = sub.add_parser("background", help="OpenAI requests in background mode: lost ones, or a whole second run")
    for x in (t, b):
        x.add_argument("map", choices=geo.MAPS)
        x.add_argument("model")
        x.add_argument("level")
    b.add_argument("--rerun", action="store_true", help="a whole second run, saved as <run>.rerun.jsonl")
    sub.add_parser("score", help="results/summary.json")
    sub.add_parser("figures", help="figures/*.png")
    a = p.parse_args(argv)

    if a.cmd == "download":
        z = zipfile.ZipFile(io.BytesIO(requests.get(DATA_URL, timeout=300).content))
        z.extractall(".")
        print("data/ ready:", len(z.namelist()), "files")
    elif a.cmd == "reference":
        from sfmap import osm
        path = run.DATA / "osm" / "sf.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(osm.fetch(path), "OpenStreetMap elements")
        s = osm.shapes(json.loads(path.read_text())["elements"])
        for key, m in geo.MAPS.items():
            report.save_reference(run.DATA, key, osm.reference(m, s), datetime.date.today().isoformat())
    elif a.cmd == "run":
        run.submit_batches(a.map)

        async def direct():
            await asyncio.gather(run.run_gemini(a.map), run.run_references(a.map))
        asyncio.run(direct())
        print("still running:", run.collect_batches())
    elif a.cmd == "references":
        asyncio.run(run.run_references(a.map, "rerun" if a.rerun else None))
    elif a.cmd == "collect":
        print("still running:", run.collect_batches())
    elif a.cmd == "retry":
        asyncio.run(run.retry_empty(a.map, a.model, a.level))
    elif a.cmd == "background":
        if a.rerun:
            run.run_openai_background(a.map, a.model, a.level, kind="rerun")
        else:
            from sfmap import score
            name = run.run_name(a.model, a.level)
            have = {r["k"] for r in score.load(run.answers_path(a.map, name))}
            missing = [k for k in range(len(geo.requests(geo.MAPS[a.map]))) if k not in have]
            print("never returned:", run.run_openai_background(a.map, a.model, a.level, ks=missing))
    elif a.cmd == "score":
        out = Path("results/summary.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report.summarize(run.DATA), indent=1))
        print("wrote", out)
    elif a.cmd == "figures":
        from sfmap import figures
        for path in figures.all_figures(run.DATA, Path("figures")):
            print("wrote", path)


if __name__ == "__main__":
    main()
