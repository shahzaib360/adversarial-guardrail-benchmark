import base64
import json
import re

import pytest

from agb.canon import canonicalize
from agb.config import load_config
from agb.data import Item
from agb.transforms import REGISTRY, build_samples, encodings, rng_for, structural

TEXT = "Explain how photosynthesis converts sunlight into chemical energy in plants."
ITEM = Item("demo:1", "harmful", "demo", 0, "test", TEXT)
DETERMINISTIC = [n for n, s in REGISTRY.items() if s.fn is not None]


def norm(s: str) -> str:
    return " ".join(s.lower().split())


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_deterministic_and_seeded(name):
    spec = REGISTRY[name]
    a = spec.fn(TEXT, rng_for(1, "demo:1", name))
    b = spec.fn(TEXT, rng_for(1, "demo:1", name))
    assert a == b
    if name != "none":
        assert a != TEXT


def test_config_transforms_are_registered():
    cfg = load_config()
    for track in cfg["tracks"].values():
        for t in track["transforms"]:
            assert t in REGISTRY, t
    assert "none" in cfg["tracks"]["harmful"]["transforms"]


def test_encoding_round_trips():
    payload = lambda s: s.split("\n\n", 1)[1]
    rng = rng_for(0, "x", "y")
    assert base64.b64decode(payload(encodings.base64_attack(TEXT, rng))).decode() == TEXT
    assert bytes.fromhex(payload(encodings.hex_attack(TEXT, rng))).decode() == TEXT
    assert encodings.rot13(payload(encodings.rot13_attack(TEXT, rng))) == TEXT
    assert payload(encodings.reverse_attack(TEXT, rng))[::-1] == TEXT
    assert encodings.from_morse(payload(encodings.morse_attack(TEXT, rng))) == TEXT.lower()


def test_payload_split_reassembles_and_breaks_words():
    out = structural.payload_split(TEXT, rng_for(0, "x", "y"))
    pieces = [json.loads(m) for m in re.findall(r'^s\d+ = (".*")$', out, flags=re.M)]
    assert "".join(pieces) == TEXT
    assert len(pieces) == 4
    # every cut falls inside a word
    for left, right in zip(pieces, pieces[1:]):
        assert not left[-1].isspace() and not right[0].isspace()


@pytest.mark.parametrize(
    "name",
    [
        "base64",
        "hex",
        "rot13",
        "morse",
        "reverse",
        "leetspeak",
        "homoglyph",
        "zero_width",
        "char_spacing",
        "fullwidth",
        "math_bold",
        "diacritics",
        "unicode_tags",
        "random_case",
    ],
)
def test_canonicalization_recovers_mechanical_attacks(name):
    attacked = REGISTRY[name].fn(TEXT, rng_for(3, "demo:1", name))
    recovered = norm(canonicalize(attacked))
    target = norm(TEXT)
    if name == "morse":  # Morse has no case and drops unsupported punctuation
        target = norm(re.sub(r"[^a-z0-9 .]", "", TEXT.lower()))
    assert target in recovered, (name, recovered)


@pytest.mark.parametrize("name", ["typos", "payload_split", "fiction_frame", "benign_padding"])
def test_canonicalization_does_not_undo_semantic_attacks(name):
    attacked = REGISTRY[name].fn(TEXT, rng_for(3, "demo:1", name))
    assert canonicalize(attacked) == attacked.strip()


def test_build_samples_skips_neural_without_backend():
    logs = []
    samples = build_samples([ITEM], ["none", "base64", "paraphrase"], seed=1, neural=None, log=logs.append)
    assert [s.transform for s in samples] == ["none", "base64"]
    assert any("paraphrase" in m for m in logs)
    assert samples[0].uid == "demo:1|none" and samples[0].family == "clean"
