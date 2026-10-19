"""Asking the models. Anthropic and OpenAI go through their Batch APIs (half price); Gemini, Jev and the Decisions
API through their regular APIs. Every request is saved as one JSON line: {k, status, text, tin, tout}.

Keys come from the environment: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY and TYPESAFE_API_KEY.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import anthropic
import httpx
import openai

from sfmap import geo, models

DATA = Path("data")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"


def run_name(model_key, level):
    return f"{model_key}-{level}"


def answers_path(map_key, name, kind=None):
    return DATA / "answers" / map_key / (f"{name}.{kind}.jsonl" if kind else f"{name}.jsonl")


def runs(map_key):
    return [(k, level) for k in models.MODELS for level in geo.MAPS[map_key].levels]


def record_from_anthropic(k, result):
    if result.get("type") != "succeeded":
        return {"k": k, "status": result.get("type"), "text": "", "tin": 0, "tout": 0}
    msg = result["message"]
    return {"k": k, "status": msg["stop_reason"],
            "text": "".join(b.get("text", "") for b in msg["content"] if b["type"] == "text"),
            "tin": msg["usage"]["input_tokens"], "tout": msg["usage"]["output_tokens"]}


def record_from_openai(k, body):
    body = body or {}
    usage = body.get("usage") or {}
    text = "".join(c.get("text", "") for item in body.get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")
    return {"k": k, "status": body.get("status"), "text": text,
            "tin": usage.get("input_tokens", 0), "tout": usage.get("output_tokens", 0)}


def record_from_gemini(k, data):
    usage = data.get("usageMetadata") or {}
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    return {"k": k, "status": cand.get("finishReason"), "text": text, "tin": usage.get("promptTokenCount", 0),
            "tout": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)}


def record_from_reference(point_id, data):
    """One Jev or Decisions API answer, with its probability for each choice. Jev gives {choice: p}, the Decisions API
    a list of {value, probability}."""
    a = data["answers"]["q0"] if isinstance(data["answers"], dict) else data["answers"][0]
    probs = a.get("probabilities")
    if isinstance(probs, list):
        probs = {x["value"]: x["probability"] for x in probs}
    return {"id": point_id, "choice": a.get("choice"), "tin": (data.get("usage") or {}).get("input_tokens", 0),
            "probs": probs}


def _append(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ---- Anthropic and OpenAI batches -------------------------------------------------------------------------------

def submit_batches(map_key):
    """One batch per model and level. Batch ids are kept in data/batches.json."""
    state_path = DATA / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    groups = geo.requests(geo.MAPS[map_key])
    ant, oai = anthropic.Anthropic(), openai.OpenAI()
    for key, level in runs(map_key):
        m, name = models.MODELS[key], f"{map_key}/{run_name(key, level)}"
        if name in state or m.provider == "gemini":
            continue
        if m.provider == "anthropic":
            reqs = [{"custom_id": f"r{k}", "params": models.anthropic_params(m, level, g)} for k, g in enumerate(groups)]
            state[name] = {"provider": "anthropic", "id": ant.messages.batches.create(requests=reqs).id}
        else:
            lines = "".join(json.dumps({"custom_id": f"r{k}", "method": "POST", "url": "/v1/responses",
                                        "body": models.openai_body(m, level, g)}) + "\n" for k, g in enumerate(groups))
            f = oai.files.create(file=(f"{map_key}-{key}-{level}.jsonl", lines.encode()), purpose="batch")
            state[name] = {"provider": "openai", "id": oai.batches.create(input_file_id=f.id, endpoint="/v1/responses",
                                                                           completion_window="24h").id}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
        print("submitted", name, flush=True)


def collect_batches(wait_s=3600):
    """Download every finished batch. Returns the names still running."""
    state = json.loads((DATA / "batches.json").read_text())
    ant, oai = anthropic.Anthropic(), openai.OpenAI()
    start = time.monotonic()
    while True:
        pending = []
        for name, b in state.items():
            map_key, run = name.split("/")
            out = answers_path(map_key, run)
            if out.exists():
                continue
            if b["provider"] == "anthropic":
                if ant.messages.batches.retrieve(b["id"]).processing_status != "ended":
                    pending.append(name)
                    continue
                recs = [record_from_anthropic(int(e.custom_id[1:]), e.result.to_dict())
                        for e in ant.messages.batches.results(b["id"])]
            else:
                ob = oai.batches.retrieve(b["id"])
                if ob.status not in ("completed", "failed", "expired", "cancelled"):
                    pending.append(name)
                    continue
                recs = []
                for line in (oai.files.content(ob.output_file_id).text.splitlines() if ob.output_file_id else []):
                    r = json.loads(line)
                    recs.append(record_from_openai(int(r["custom_id"][1:]), (r.get("response") or {}).get("body")))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in sorted(recs, key=lambda r: r["k"])))
            print("collected", name, len(recs), flush=True)
        if not pending or time.monotonic() - start > wait_s:
            return pending
        time.sleep(30)


# ---- regular APIs ------------------------------------------------------------------------------------------------

async def _post(client, url, **kw):
    """POST with waits on rate limits and server errors (an empty answer is not an error)."""
    r = None
    for attempt in range(8):
        try:
            r = await client.post(url, **kw)
        except httpx.HTTPError:
            await asyncio.sleep(min(60, 2 ** attempt))
            continue
        if r.status_code not in (429, 500, 502, 503, 504):
            return r
        await asyncio.sleep(min(60, 2 ** attempt))
    return r


async def ask_gemini(client, m, level, group):
    r = await _post(client, GEMINI_URL.format(m.model_id), params={"key": os.environ["GEMINI_API_KEY"]},
                    json=models.gemini_body(m, level, group))
    return r.json() if r is not None and r.status_code == 200 else {}


async def run_gemini(map_key, concurrency=64):
    """Every Gemini model and level of a map, one try per request."""
    groups = geo.requests(geo.MAPS[map_key])
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=1200) as client:
        async def one(key, level, k):
            async with sem:
                data = await ask_gemini(client, models.MODELS[key], level, groups[k])
            _append(answers_path(map_key, run_name(key, level), "part"), record_from_gemini(k, data))
        jobs = []
        for key, level in runs(map_key):
            if models.MODELS[key].provider != "gemini" or answers_path(map_key, run_name(key, level)).exists():
                continue
            part = answers_path(map_key, run_name(key, level), "part")
            done = {json.loads(x)["k"] for x in part.read_text().splitlines()} if part.exists() else set()
            jobs += [one(key, level, k) for k in range(len(groups)) if k not in done]
        await asyncio.gather(*jobs)
    for key, level in runs(map_key):
        part = answers_path(map_key, run_name(key, level), "part")
        if part.exists() and len(part.read_text().splitlines()) >= len(groups):
            part.rename(answers_path(map_key, run_name(key, level)))


async def retry_empty(map_key, key, level, extra=4):
    """Ask again, up to `extra` more times, every Gemini request of a run that came back empty or incomplete.
    The tries go to <run>.retries.jsonl; scoring uses the first complete one."""
    from sfmap import prompt, score
    m, name = models.MODELS[key], run_name(key, level)
    groups = geo.requests(geo.MAPS[map_key])
    retries_path = answers_path(map_key, name, "retries")
    tried = {}
    for r in score.load(retries_path):
        tried.setdefault(r["k"], []).append(r)
    todo = [r["k"] for r in score.load(answers_path(map_key, name))
            if len(prompt.parse(r["text"], len(groups[r["k"]]))) < len(groups[r["k"]])]

    async with httpx.AsyncClient(timeout=1200) as client:
        async def one(k):
            recs = tried.get(k, [])
            while len(recs) < extra and not any(len(prompt.parse(x["text"], len(groups[k]))) == len(groups[k])
                                                for x in recs):
                rec = record_from_gemini(k, await ask_gemini(client, m, level, groups[k])) | {"attempt": len(recs) + 2}
                _append(retries_path, rec)
                recs.append(rec)
        await asyncio.gather(*(one(k) for k in todo))


def run_openai_background(map_key, key, level, ks=None, kind=None, wait_s=7200):
    """Requests through OpenAI's regular API in background mode: the request runs on OpenAI's side and we poll for
    it, so long reasoning (20-30 minutes at high effort) cannot time out. Used for requests a stalled batch never
    returned (kind=None appends them to the run, at the standard price) and for second runs (kind="rerun")."""
    m, name = models.MODELS[key], run_name(key, level)
    groups = geo.requests(geo.MAPS[map_key])
    ks = range(len(groups)) if ks is None else ks
    client = openai.OpenAI()
    ids = {k: client.responses.create(**models.openai_body(m, level, groups[k]), background=True).id for k in ks}
    start = time.monotonic()
    while ids and time.monotonic() - start < wait_s:
        for k, rid in list(ids.items()):
            r = client.responses.retrieve(rid)
            if r.status in ("queued", "in_progress"):
                continue
            _append(answers_path(map_key, name, kind), record_from_openai(k, r.to_dict()) | {"priced": "standard"})
            del ids[k]
        time.sleep(20)
    return sorted(ids)


async def run_references(map_key, kind=None):
    """Jev and the Decisions API, one point per request. kind="rerun" saves a whole second run as <name>.rerun.jsonl."""
    pts, _, _ = geo.points(geo.MAPS[map_key])
    targets = {"jev": ("https://api.typesafe.ai/v1/systemone", os.environ.get("TYPESAFE_API_KEY"), models.jev_body, 32),
               "decisions": ("https://api.openai.com/v1/decisions", os.environ.get("OPENAI_API_KEY"),
                             models.decisions_body, 48)}
    async with httpx.AsyncClient(timeout=60) as client:
        async def one(ref, p, sem):
            url, key, body, _ = targets[ref]
            async with sem:
                r = await _post(client, url, headers={"Authorization": f"Bearer {key}"}, json=body(p))
            if r is None or r.status_code != 200:
                return
            _append(answers_path(map_key, ref, kind), record_from_reference(p["id"], r.json()))
        jobs = []
        for ref, (_, _, _, limit) in targets.items():
            path = answers_path(map_key, ref, kind)
            done = {json.loads(x)["id"] for x in path.read_text().splitlines()} if path.exists() else set()
            sem = asyncio.Semaphore(limit)
            jobs += [one(ref, p, sem) for p in pts if p["id"] not in done]
        await asyncio.gather(*jobs)
