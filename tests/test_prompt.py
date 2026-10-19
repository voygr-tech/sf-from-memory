import json

from sfmap import prompt

TWO = [{"lat": 37.80241, "lon": -122.44853}, {"lat": 37.76944, "lon": -122.48622}]


def test_build_lists_points_in_order_with_five_decimals():
    text = prompt.build(TWO)
    assert "1. 37.80241, -122.44853\n2. 37.76944, -122.48622" in text


def test_build_says_how_many_points_and_asks_for_numbered_letters():
    text = prompt.build(TWO)
    assert "Below are 2 locations in San Francisco, in no particular order," in text
    assert 'one entry per location, written as its number, a colon and its letter, from "1:H" to "2:S"' in text


def test_build_single_point_uses_singular():
    assert "Below is 1 location as WGS84" in prompt.build(TWO[:1])


def test_build_defines_all_five_letters():
    text = prompt.build(TWO)
    for letter in ("W = water", "P = park", "H = inside a building", "S = on a street", "O = anything else"):
        assert letter in text


def test_parse_numbered_answers():
    text = json.dumps({"codes": ["1:H", "2:P"]})
    assert prompt.parse(text, 2) == {0: "building", 1: "park"}


def test_parse_numbered_answers_in_any_order_with_spaces_and_lower_case():
    text = json.dumps({"codes": [" 2 : w", "1:s"]})
    assert prompt.parse(text, 2) == {0: "street", 1: "water"}


def test_parse_numbered_skips_missing_unknown_and_out_of_range_entries():
    text = json.dumps({"codes": ["1:H", "2:X", "3:P", "x:W"]})
    assert prompt.parse(text, 2) == {0: "building"}


def test_parse_plain_letters_when_count_matches():
    assert prompt.parse(json.dumps({"codes": ["O", "W"]}), 2) == {0: "other", 1: "water"}


def test_parse_plain_letters_with_wrong_count_gives_nothing():
    assert prompt.parse(json.dumps({"codes": ["O", "W", "P"]}), 2) == {}


def test_parse_bad_json_or_missing_key_gives_nothing():
    assert prompt.parse("not json", 2) == {}
    assert prompt.parse(json.dumps({"answers": ["1:H"]}), 2) == {}
    assert prompt.parse("", 2) == {}


def test_letter_classes_are_the_five_classes():
    assert sorted(prompt.LETTERS.values()) == sorted(prompt.CLASSES)
