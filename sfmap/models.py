"""The models in the benchmark, what each request looks like, and what it costs."""

from dataclasses import dataclass

from sfmap import prompt


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    provider: str  # anthropic, openai or gemini
    model_id: str
    price_in: float  # $ per 1M input tokens, standard price
    price_out: float  # $ per 1M output tokens, reasoning included
    batch: bool  # sent through the provider's Batch API, at half price


MODELS = {
    "fable": Model("fable", "Claude Fable 5.1", "anthropic", "claude-fable-5-1", 10, 50, True),
    "opus": Model("opus", "Claude Opus 5.5", "anthropic", "claude-opus-5-5", 4, 20, True),
    "astra": Model("astra", "GPT-6 Astra", "openai", "gpt-6-astra", 10, 50, True),
    "sol": Model("sol", "GPT-6.1 Sol", "openai", "gpt-6.1-sol", 2, 10, True),
    "argon": Model("argon", "Gemini 4 Argon", "gemini", "gemini-4-argon", 2, 10, False),
    "flash": Model("flash", "Gemini 3.8 Flash", "gemini", "gemini-3.8-flash", 0.75, 3.75, False),
}

# Two point-at-a-time references: TypeSafe's Jev and OpenAI's Decisions API (GPT-6 Luna). $ per 1M input tokens.
REFERENCES = {"jev": ("Jev", 0.042), "decisions": ("OpenAI Decisions API", 0.10)}

JEV_CRITERIA = {"water": "Bay, ocean, lake or pond.", "park": "Park, lawn, trees or other green open space.",
                "building": "Inside a building.", "street": "On a street (roadway or sidewalk).",
                "other": "Anything else, such as a parking lot, plaza, beach or empty lot."}


def anthropic_params(model, level, group):
    return {"model": model.model_id, "max_tokens": 128_000 if model.key == "fable" else 64_000,
            "output_config": {"effort": level, "format": {"type": "json_schema", "schema": prompt.SCHEMA}},
            "messages": [{"role": "user", "content": prompt.build(group)}]}


def openai_body(model, level, group):
    return {"model": model.model_id, "input": prompt.build(group), "reasoning": {"effort": level},
            "max_output_tokens": 100_000,
            "text": {"format": {"type": "json_schema", "name": "answer", "schema": prompt.SCHEMA, "strict": True}}}


def gemini_body(model, level, group):
    return {"contents": [{"role": "user", "parts": [{"text": prompt.build(group)}]}],
            "generationConfig": {"thinkingConfig": {"thinkingLevel": level}, "responseMimeType": "application/json",
                                 "responseSchema": prompt.GEMINI_SCHEMA,
                                 "maxOutputTokens": 32_768 if model.key == "argon" else 65_536}}


def jev_body(point):
    return {"state": "San Francisco, California. Answer from your own knowledge only.", "model": "jev-latest",
            "questions": {"q0": {"type": "choice", "criteria": JEV_CRITERIA,
                                 "instructions": f"What is at latitude {point['lat']:.6f}, longitude {point['lon']:.6f} "
                                                 "in San Francisco?"}}}


def decisions_body(point):
    return {"model": "gpt-6-luna",
            "input": f"San Francisco, California. Location (WGS84): latitude {point['lat']:.6f}, "
                     f"longitude {point['lon']:.6f}. Answer from your own knowledge only.",
            "questions": [{"type": "choice", "name": "land_use", "instructions": "What is at this exact spot?",
                           "choices": [{"value": k, "description": v} for k, v in JEV_CRITERIA.items()]}]}


def cost(model, tokens_in, tokens_out, batch):
    return (tokens_in * model.price_in + tokens_out * model.price_out) / 1e6 * (0.5 if batch else 1)
