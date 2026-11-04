"""Unicode and character-level obfuscations.

These leave the text readable to a human (or to an LLM) but change the byte/token
sequence the guardrail sees.
"""

from __future__ import annotations

import random
import re

# Latin -> visually identical Cyrillic code points.
HOMOGLYPHS = {
    "a": "\u0430", "c": "\u0441", "e": "\u0435", "o": "\u043e", "p": "\u0440", "x": "\u0445",
    "y": "\u0443", "i": "\u0456", "j": "\u0458", "s": "\u0455",
    "A": "\u0410", "B": "\u0412", "C": "\u0421", "E": "\u0415", "H": "\u041d", "I": "\u0406",
    "J": "\u0408", "K": "\u041a", "M": "\u041c", "O": "\u041e", "P": "\u0420", "S": "\u0405",
    "T": "\u0422", "X": "\u0425", "Y": "\u04ae",
}  # fmt: skip
_HOMOGLYPH_TABLE = str.maketrans(HOMOGLYPHS)


def homoglyph(text: str, rng: random.Random) -> str:
    return text.translate(_HOMOGLYPH_TABLE)


ZERO_WIDTH_SPACE = "\u200b"


def zero_width(text: str, rng: random.Random) -> str:
    """Insert a zero-width space between every pair of characters."""
    return ZERO_WIDTH_SPACE.join(text)


def char_spacing(text: str, rng: random.Random) -> str:
    """'how to' -> 'h o w   t o' (the spacing trick publicly reported against Prompt-Guard-86M)."""
    return "\n".join("   ".join(" ".join(word) for word in line.split()) for line in text.split("\n"))


def fullwidth(text: str, rng: random.Random) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in text)


def math_bold(text: str, rng: random.Random) -> str:
    """Map ASCII letters/digits to the Mathematical Bold block (U+1D400...)."""
    out = []
    for c in text:
        o = ord(c)
        if 0x41 <= o <= 0x5A:
            out.append(chr(0x1D400 + o - 0x41))
        elif 0x61 <= o <= 0x7A:
            out.append(chr(0x1D41A + o - 0x61))
        elif 0x30 <= o <= 0x39:
            out.append(chr(0x1D7CE + o - 0x30))
        else:
            out.append(c)
    return "".join(out)


_MARKS = ["\u0301", "\u0300", "\u0308", "\u0303", "\u0302", "\u030c"]


def diacritics(text: str, rng: random.Random) -> str:
    """Attach one combining mark to every ASCII letter (light 'Zalgo' text)."""
    out, k = [], 0
    for c in text:
        out.append(c)
        if c.isascii() and c.isalpha():
            out.append(_MARKS[k % len(_MARKS)])
            k += 1
    return "".join(out)


def to_tags(text: str) -> str:
    """ASCII -> invisible Unicode tag characters (U+E0020..U+E007E)."""
    return "".join(chr(0xE0000 + ord(c)) if 0x20 <= ord(c) <= 0x7E else c for c in text)


def unicode_tags(text: str, rng: random.Random) -> str:
    """'ASCII smuggling': an innocuous visible greeting followed by the payload in invisible tag characters."""
    return "Hello!" + to_tags(text)


def random_case(text: str, rng: random.Random) -> str:
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)


_KEYBOARD = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok",
    "o": "ipl", "p": "ol", "a": "qsz", "s": "awdz", "d": "sefx", "f": "drgc", "g": "fthv", "h": "gyjb",
    "j": "hukn", "k": "jilm", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}  # fmt: skip


def typos(text: str, rng: random.Random, rate: float = 0.35) -> str:
    """Natural-looking typing noise: swap / drop / duplicate / keyboard-neighbour substitution."""

    def mutate(word: str) -> str:
        letters = [i for i, c in enumerate(word) if c.isalpha()]
        if len(letters) < 4 or rng.random() >= rate:
            return word
        i = rng.choice(letters[1:-1])
        op = rng.choice(("swap", "drop", "dup", "sub"))
        if op == "swap" and i + 1 < len(word) and word[i + 1].isalpha():
            return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if op == "drop":
            return word[:i] + word[i + 1 :]
        if op == "dup":
            return word[:i] + word[i] + word[i:]
        neighbours = _KEYBOARD.get(word[i].lower())
        if neighbours:
            r = rng.choice(neighbours)
            return word[:i] + (r.upper() if word[i].isupper() else r) + word[i + 1 :]
        return word

    return re.sub(r"\S+", lambda m: mutate(m.group(0)), text)
