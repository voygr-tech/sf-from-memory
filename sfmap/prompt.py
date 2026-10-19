"""The question every model gets, and how its answer is read."""

import json

CLASSES = ["water", "park", "building", "street", "other"]
LETTERS = {"W": "water", "P": "park", "H": "building", "S": "street", "O": "other"}

TEMPLATE = """You are taking part in a map-knowledge test about San Francisco, California. Answer from your own knowledge only.

Below {what} as WGS84 latitude, longitude:

{lines}

For each location give one letter for what is at that exact spot: W = water (bay, ocean, lake or pond), P = park, lawn, trees or other green open space, H = inside a building, S = on a street (roadway or sidewalk), O = anything else (parking lot, plaza, beach or empty lot).

Reply with JSON only: "codes", one entry per location, written as its number, a colon and its letter, from "1:H" to "{n}:S" (example letters)."""

# The JSON shape the answer must have (Anthropic and OpenAI structured outputs, and Gemini's own schema dialect).
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["codes"],
          "properties": {"codes": {"type": "array", "items": {"type": "string"}}}}
GEMINI_SCHEMA = {"type": "OBJECT", "properties": {"codes": {"type": "ARRAY", "items": {"type": "STRING"}}},
                 "required": ["codes"]}


def build(group):
    n = len(group)
    what = "is 1 location" if n == 1 else f"are {n} locations in San Francisco, in no particular order,"
    lines = "\n".join(f"{i + 1}. {p['lat']:.5f}, {p['lon']:.5f}" for i, p in enumerate(group))
    return TEMPLATE.format(what=what, lines=lines, n=n)


def parse(text, n):
    """{index in the request: class} from numbered ("17:H") or plain letter answers. Anything missing or
    unreadable is left out, so it counts as no answer."""
    try:
        codes = json.loads(text)["codes"]
    except (ValueError, KeyError, TypeError):
        return {}
    out = {}
    if codes and all(isinstance(c, str) and ":" in c for c in codes):
        for c in codes:
            num, _, letter = c.partition(":")
            try:
                i = int(num.strip()) - 1
            except ValueError:
                continue
            cls = LETTERS.get(letter.strip().upper()[:1])
            if 0 <= i < n and cls:
                out[i] = cls
    elif len(codes) == n:
        for i, c in enumerate(codes):
            cls = LETTERS.get(str(c).strip().upper()[:1])
            if cls:
                out[i] = cls
    return out
