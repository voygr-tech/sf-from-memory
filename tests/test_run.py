from sfmap import run


def test_record_from_anthropic_success():
    result = {"type": "succeeded", "message": {
        "stop_reason": "end_turn", "usage": {"input_tokens": 1800, "output_tokens": 950},
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": '{"codes": ["1:H"]}'}]}}
    assert run.record_from_anthropic(7, result) == {"k": 7, "status": "end_turn", "text": '{"codes": ["1:H"]}',
                                                    "tin": 1800, "tout": 950}


def test_record_from_anthropic_error_is_an_empty_answer():
    assert run.record_from_anthropic(2, {"type": "errored"}) == {"k": 2, "status": "errored", "text": "",
                                                                 "tin": 0, "tout": 0}


def test_record_from_openai_takes_only_the_message_text():
    body = {"status": "completed", "usage": {"input_tokens": 1500, "output_tokens": 4000},
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": '{"codes": []}'}]}]}
    assert run.record_from_openai(3, body) == {"k": 3, "status": "completed", "text": '{"codes": []}',
                                               "tin": 1500, "tout": 4000}


def test_record_from_openai_missing_body_is_an_empty_answer():
    assert run.record_from_openai(1, None) == {"k": 1, "status": None, "text": "", "tin": 0, "tout": 0}


def test_record_from_gemini_counts_thinking_as_output_and_drops_thought_text():
    data = {"usageMetadata": {"promptTokenCount": 2500, "candidatesTokenCount": 320, "thoughtsTokenCount": 9000},
            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"text": "thinking...", "thought": True}, {"text": '{"codes": ["1:W"]}'}]}}]}
    assert run.record_from_gemini(5, data) == {"k": 5, "status": "STOP", "text": '{"codes": ["1:W"]}',
                                               "tin": 2500, "tout": 9320}


def test_record_from_gemini_out_of_room_has_no_text():
    data = {"usageMetadata": {"promptTokenCount": 2500, "candidatesTokenCount": 3, "thoughtsTokenCount": 32765},
            "candidates": [{"finishReason": "MAX_TOKENS", "content": {}}]}
    rec = run.record_from_gemini(0, data)
    assert rec["status"] == "MAX_TOKENS" and rec["text"] == "" and rec["tout"] == 32768


def test_run_name_and_paths():
    assert run.run_name("sol", "high") == "sol-high"
    p = run.answers_path("sf100", "sol-high")
    assert p.parts[-2:] == ("sf100", "sol-high.jsonl")
    assert run.answers_path("sf100", "sol-high", "retries").name == "sol-high.retries.jsonl"


def test_runs_of_a_map_are_every_model_at_every_level():
    names = run.runs("sf100")
    assert len(names) == 18 and ("fable", "low") in names and ("flash", "high") in names
    assert len(run.runs("marina10")) == 6


JEV_REPLY = {"answers": {"q0": {"type": "choice", "choice": "water", "confidence": 0.2,
                                "probabilities": {"water": 0.36, "park": 0.17, "building": 0.27, "street": 0.15,
                                                  "other": 0.05}}},
             "usage": {"input_tokens": 423, "output_tokens": 53}}


def test_record_from_reference_keeps_jev_probabilities():
    assert run.record_from_reference("060:120", JEV_REPLY) == {
        "id": "060:120", "choice": "water", "tin": 423,
        "probs": {"water": 0.36, "park": 0.17, "building": 0.27, "street": 0.15, "other": 0.05}}


def test_record_from_reference_turns_the_decisions_probability_list_into_a_dict():
    reply = {"answers": [{"type": "choice", "name": "land_use", "choice": "park",
                          "probabilities": [{"value": "water", "probability": 0.29},
                                            {"value": "park", "probability": 0.71}]}],
             "usage": {"input_tokens": 200}}
    rec = run.record_from_reference("000:001", reply)
    assert rec == {"id": "000:001", "choice": "park", "tin": 200, "probs": {"water": 0.29, "park": 0.71}}


def test_record_from_reference_without_probabilities_or_usage():
    rec = run.record_from_reference("000:001", {"answers": {"q0": {"choice": "water"}}})
    assert rec == {"id": "000:001", "choice": "water", "tin": 0, "probs": None}
