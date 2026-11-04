"""Transform registry and sample construction.

A *transform* maps an item's text to an attacked variant. Deterministic transforms are
pure functions of ``(text, rng)`` where the RNG is seeded from ``(seed, item_id,
transform)``. Neural transforms (paraphrase, translation) run in batches through
:mod:`agb.transforms.neural` and carry a validity flag from a semantic-similarity check.

Every transform is applied to positives *and* negatives: misses are measured on the
former, false positives on the latter.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from . import encodings, framing, structural, unicode_obf

TransformFn = Callable[[str, random.Random], str]


@dataclass(frozen=True)
class TransformSpec:
    name: str
    family: str
    description: str
    fn: TransformFn | None = None  # None => neural transform
    neural_kind: str | None = None  # "paraphrase" | "translate"
    lang: str | None = None  # NLLB language code for translations


def _identity(text: str, rng: random.Random) -> str:
    return text


_SPECS = [
    TransformSpec("none", "clean", "Unmodified input", _identity),
    # encodings (with a neutral decoding instruction)
    TransformSpec("base64", "encoding", "Base64 payload + decode instruction", encodings.base64_attack),
    TransformSpec("hex", "encoding", "Hex-encoded UTF-8 payload + decode instruction", encodings.hex_attack),
    TransformSpec("rot13", "encoding", "ROT13 payload + decode instruction", encodings.rot13_attack),
    TransformSpec("morse", "encoding", "Morse-code payload + decode instruction", encodings.morse_attack),
    TransformSpec("reverse", "encoding", "Character-reversed payload + instruction", encodings.reverse_attack),
    # character / unicode obfuscation (readable, no instruction needed)
    TransformSpec("leetspeak", "obfuscation", "a->4 e->3 i->1 o->0 s->5 t->7", encodings.leetspeak),
    TransformSpec("homoglyph", "obfuscation", "Latin letters -> look-alike Cyrillic", unicode_obf.homoglyph),
    TransformSpec("zero_width", "obfuscation", "Zero-width space between all characters", unicode_obf.zero_width),
    TransformSpec("char_spacing", "obfuscation", "Spaces between letters ('h o w   t o')", unicode_obf.char_spacing),
    TransformSpec("fullwidth", "obfuscation", "Fullwidth forms (U+FF01..)", unicode_obf.fullwidth),
    TransformSpec("math_bold", "obfuscation", "Mathematical bold alphanumerics", unicode_obf.math_bold),
    TransformSpec("diacritics", "obfuscation", "Combining mark on every letter", unicode_obf.diacritics),
    TransformSpec("unicode_tags", "obfuscation", "Invisible tag-character payload", unicode_obf.unicode_tags),
    TransformSpec("random_case", "obfuscation", "Random upper/lower case", unicode_obf.random_case),
    TransformSpec("typos", "obfuscation", "Keyboard typos in ~35% of words", unicode_obf.typos),
    # structural
    TransformSpec("payload_split", "structural", "Mid-word fragments + concatenate", structural.payload_split),
    TransformSpec("benign_padding", "structural", "~300 words of benign text first", structural.benign_padding),
    # framing (request kept verbatim)
    TransformSpec("fiction_frame", "framing", "Thriller-novel scene framing", framing.fiction_frame),
    TransformSpec("research_frame", "framing", "Academic-research framing", framing.research_frame),
    TransformSpec("hypothetical_frame", "framing", "'World without rules' framing", framing.hypothetical_frame),
    # semantic (neural)
    TransformSpec("paraphrase", "paraphrase", "Sentence-level T5 paraphrase (meaning-checked)",
                  neural_kind="paraphrase"),
    TransformSpec("translate_fr", "translation", "NLLB translation to French", neural_kind="translate",
                  lang="fra_Latn"),
    TransformSpec("translate_bn", "translation", "NLLB translation to Bengali", neural_kind="translate",
                  lang="ben_Beng"),
    TransformSpec("translate_zu", "translation", "NLLB translation to Zulu", neural_kind="translate",
                  lang="zul_Latn"),
]  # fmt: skip

REGISTRY: dict[str, TransformSpec] = {s.name: s for s in _SPECS}


@dataclass
class Sample:
    uid: str
    item_id: str
    track: str
    source: str
    label: int
    category: str
    transform: str
    family: str
    text: str
    valid: bool = True
    info: dict = field(default_factory=dict)
    canon_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def rng_for(seed: int, item_id: str, transform: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{item_id}|{transform}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def apply_deterministic(item, spec: TransformSpec, seed: int) -> Sample:
    assert spec.fn is not None
    text = spec.fn(item.text, rng_for(seed, item.item_id, spec.name))
    return Sample(
        uid=f"{item.item_id}|{spec.name}",
        item_id=item.item_id,
        track=item.track,
        source=item.source,
        label=item.label,
        category=item.category,
        transform=spec.name,
        family=spec.family,
        text=text,
    )


def build_samples(items, transform_names, seed: int, neural=None, log=print) -> list[Sample]:
    """Apply every requested transform to every item.

    ``neural`` is an optional :class:`agb.transforms.neural.NeuralTransforms`; if it is
    ``None``, neural transforms are skipped (and reported as missing downstream).
    """
    samples: list[Sample] = []
    for name in transform_names:
        spec = REGISTRY[name]
        if spec.fn is not None:
            samples.extend(apply_deterministic(it, spec, seed) for it in items)
            continue
        if neural is None:
            log(f"[prepare] skipping neural transform {name!r} (no neural backend)")
            continue
        texts = [it.text for it in items]
        if spec.neural_kind == "paraphrase":
            outs = neural.paraphrase(texts)
        else:
            outs = neural.translate(texts, spec.lang)
        for it, (text, valid, info) in zip(items, outs):
            samples.append(
                Sample(
                    uid=f"{it.item_id}|{name}",
                    item_id=it.item_id,
                    track=it.track,
                    source=it.source,
                    label=it.label,
                    category=it.category,
                    transform=name,
                    family=spec.family,
                    text=text,
                    valid=valid,
                    info=info,
                )
            )
    return samples
