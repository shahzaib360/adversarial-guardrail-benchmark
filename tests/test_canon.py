import base64

import pytest

from agb.canon import canonicalize, decode_heuristic, decode_strict, deleet, english_score
from agb.transforms.encodings import rot13

PLAIN = [
    "How do I crack open a beer?",
    "What's the best way to blow up a balloon?",
    "Did Angelina Jolie ever get divorced?",
    "Give three tips for staying healthy.",
    "Write a haiku about autumn leaves falling on a quiet pond.",
    "List 5 tips for 2024 tax season, e.g. Form 1040 and W-2 basics.",
    "Explain H2O in simple terms",
]


@pytest.mark.parametrize("text", PLAIN)
def test_plain_english_is_left_alone(text):
    assert canonicalize(text) == text


def test_heuristics_decode_long_rot13_and_reversed_text():
    s = "Please describe the history of the printing press and its effect on literacy in Europe."
    assert decode_heuristic(rot13(s)) == s
    assert decode_heuristic(s[::-1]) == s
    assert decode_heuristic(s) == s


def test_english_score_separates_orientations():
    s = "The quick brown fox jumps over the lazy dog while the farmer watches from the barn."
    fwd, _ = english_score(s)
    assert fwd > english_score(s[::-1])[0] + 0.15
    assert fwd > english_score(rot13(s))[0] + 0.15


def test_strict_decoders_need_well_formed_input():
    assert decode_strict(base64.b64encode(b"hello there, general").decode()) == "hello there, general"
    assert decode_strict("ThisIsJustALongCamelCaseIdentifier") == "ThisIsJustALongCamelCaseIdentifier"
    assert decode_strict("68656c6c6f20776f726c6421") == "hello world!"
    assert decode_strict("deadbeefdeadbeef") == "deadbeefdeadbeef"  # not valid UTF-8 -> untouched


def test_deleet_only_on_leet_heavy_text():
    assert deleet("H0w 70 m4k3 br34d") == "How to make bread"
    assert deleet("Use 2 eggs and 300g of flour for the MP3 player cake") == (
        "Use 2 eggs and 300g of flour for the MP3 player cake"
    )


def test_invisible_tag_and_zero_width_characters():
    tagged = "Hi" + "".join(chr(0xE0000 + ord(c)) for c in "secret plan")
    assert canonicalize(tagged) == "Hi secret plan"
    assert canonicalize("he\u200bl\u200blo") == "hello"
