import pytest

from sfmap import models, prompt

GROUP = [{"lat": 37.80241, "lon": -122.44853}, {"lat": 37.76944, "lon": -122.48622}]


def test_every_model_has_a_provider_we_can_call():
    assert {m.provider for m in models.MODELS.values()} == {"anthropic", "openai", "gemini"}


def test_two_models_per_provider():
    per = {}
    for m in models.MODELS.values():
        per[m.provider] = per.get(m.provider, 0) + 1
    assert per == {"anthropic": 2, "openai": 2, "gemini": 2}


def test_anthropic_params_sets_effort_schema_and_prompt():
    p = models.anthropic_params(models.MODELS["opus"], "high", GROUP)
    assert p["model"] == "claude-opus-5-5"
    assert p["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": prompt.SCHEMA}}
    assert p["messages"] == [{"role": "user", "content": prompt.build(GROUP)}]


def test_anthropic_params_gives_fable_room_for_long_reasoning():
    assert models.anthropic_params(models.MODELS["fable"], "high", GROUP)["max_tokens"] == 128_000


def test_openai_body_sets_reasoning_effort_and_strict_schema():
    b = models.openai_body(models.MODELS["sol"], "medium", GROUP)
    assert b["model"] == "gpt-6.1-sol"
    assert b["reasoning"] == {"effort": "medium"}
    assert b["text"]["format"] == {"type": "json_schema", "name": "answer", "schema": prompt.SCHEMA, "strict": True}
    assert b["input"] == prompt.build(GROUP)


def test_gemini_body_sets_thinking_level_and_argon_output_cap():
    b = models.gemini_body(models.MODELS["argon"], "low", GROUP)
    assert b["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert b["generationConfig"]["maxOutputTokens"] == 32_768
    assert b["generationConfig"]["responseSchema"] == prompt.GEMINI_SCHEMA
    assert models.gemini_body(models.MODELS["flash"], "low", GROUP)["generationConfig"]["maxOutputTokens"] == 65_536


def test_jev_body_asks_one_choice_question_about_one_point():
    b = models.jev_body(GROUP[0])
    assert b["model"] == "jev-latest"
    q = b["questions"]["q0"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"water", "park", "building", "street", "other"}
    assert "latitude 37.802410, longitude -122.448530" in q["instructions"]


def test_decisions_body_uses_luna_and_five_choices():
    b = models.decisions_body(GROUP[1])
    assert b["model"] == "gpt-6-luna"
    assert [c["value"] for c in b["questions"][0]["choices"]] == ["water", "park", "building", "street", "other"]
    assert "latitude 37.769440, longitude -122.486220" in b["input"]


def test_cost_batch_runs_pay_half():
    m = models.MODELS["astra"]  # $10 / $50 per 1M tokens
    assert models.cost(m, 1_000_000, 1_000_000, batch=True) == pytest.approx(30.0)
    assert models.cost(m, 1_000_000, 1_000_000, batch=False) == pytest.approx(60.0)


def test_cost_zero_tokens_is_free():
    assert models.cost(models.MODELS["flash"], 0, 0, batch=False) == 0


def test_batch_discount_only_for_anthropic_and_openai():
    assert models.MODELS["fable"].batch and models.MODELS["sol"].batch
    assert not models.MODELS["argon"].batch
