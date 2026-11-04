"""Input canonicalization: a cheap, model-agnostic pre-processing defense.

``canonicalize(text)`` produces a normalised *view* of the input:

1. decode invisible Unicode tag characters back to ASCII ("ASCII smuggling"),
2. strip zero-width / invisible / bidi-control characters,
3. NFKC-normalise (fullwidth forms, mathematical alphanumerics -> ASCII),
4. strip combining marks attached to Latin letters,
5. map Cyrillic/Greek look-alikes to Latin in Latin-dominated text,
6. collapse letter-spaced text ("h o w   t o" -> "how to"),
7. decode Base64 / hex / Morse segments (strict format checks),
8. undo leetspeak in text dominated by mixed letter/digit tokens,
9. undo ROT13 / character reversal when the result looks markedly more like English
   (share of the most frequent English letter bigrams),
10. lower-case text with erratic capitalisation.

The defended classifier scores ``max(score(text), score(canonicalize(text)))``, so
canonicalization can only add detections -- the price is paid in false positives, which
the benchmark measures on canonicalized benign inputs.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata

from .transforms.encodings import from_morse, rot13

_TAG_RUN = re.compile("[\U000e0000-\U000e007f]+")
_INVISIBLE = re.compile(
    "[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f\u202a-\u202e"
    "\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\U0001d173-\U0001d17a\U000e0100-\U000e01ef]"
)

CONFUSABLES = {
    # Cyrillic
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0445": "x", "\u0443": "y",
    "\u0456": "i", "\u0458": "j", "\u0455": "s", "\u04bb": "h", "\u0501": "d", "\u051b": "q", "\u051d": "w",
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E", "\u041d": "H", "\u0406": "I", "\u0408": "J",
    "\u041a": "K", "\u041c": "M", "\u041e": "O", "\u0420": "P", "\u0405": "S", "\u0422": "T", "\u0425": "X",
    "\u04ae": "Y", "\u0423": "Y",
    # Greek
    "\u03b1": "a", "\u03bf": "o", "\u03c1": "p", "\u03b9": "i", "\u03ba": "k", "\u03bd": "v", "\u03c5": "u",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H", "\u0399": "I", "\u039a": "K",
    "\u039c": "M", "\u039d": "N", "\u039f": "O", "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X",
}  # fmt: skip
_CONFUSABLE_TABLE = str.maketrans(CONFUSABLES)

_LEET_BACK = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"})

_B64 = re.compile(r"^[A-Za-z0-9+/]{16,}={0,2}$")
_HEX = re.compile(r"^(?:[0-9a-fA-F]{2}){8,}$")
_MORSE = re.compile(r"^[.\-/ ]+$")
_LETTER_RUN = re.compile(r"[a-z]+")
# The 42 most frequent English letter bigrams (Norvig's Google Books counts); together they
# cover roughly half of all letter pairs in English text, but only ~15% of reversed English
# and far less of ROT13 text.
_TOP_BIGRAMS = frozenset(
    "th he in er an re on at en nd ti es or te of ed is it al ar st to nt ng se ha as ou io le ve co me de hi "
    "ri ro ic ne ea ra ce".split()
)


def english_score(text: str) -> tuple[float, int]:
    """(share of within-word letter pairs that are top English bigrams, number of pairs)."""
    pairs = hits = 0
    for word in _LETTER_RUN.findall(text.lower()):
        for a, b in zip(word, word[1:]):
            pairs += 1
            hits += (a + b) in _TOP_BIGRAMS
    return (hits / pairs if pairs else 0.0), pairs


def decode_tags(text: str) -> str:
    def repl(m: re.Match) -> str:
        decoded = "".join(chr(ord(c) - 0xE0000) for c in m.group(0) if 0xE0020 <= ord(c) <= 0xE007E)
        return f" {decoded} " if decoded else ""

    return _TAG_RUN.sub(repl, text)


def strip_latin_marks(text: str) -> str:
    out: list[str] = []
    prev_latin = False
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) == "Mn":
            if prev_latin:
                continue
        else:
            prev_latin = ch.isascii() and ch.isalpha()
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _script_counts(text: str) -> tuple[int, int]:
    latin = other = 0
    for ch in text:
        if ch.isascii() and ch.isalpha():
            latin += 1
        elif ch in CONFUSABLES:
            other += 1
    return latin, other


def map_confusables(text: str) -> str:
    latin, other = _script_counts(text)
    if other == 0:
        return text
    if latin >= 0.3 * (latin + other):
        return text.translate(_CONFUSABLE_TABLE)
    # Genuinely non-Latin text: only fix words that mix scripts.
    return re.sub(
        r"\S+",
        lambda m: (
            m.group(0).translate(_CONFUSABLE_TABLE)
            if any(c.isascii() and c.isalpha() for c in m.group(0))
            else m.group(0)
        ),
        text,
    )


def collapse_spacing(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        toks = [t for t in line.split(" ") if t]
        if len(toks) >= 6 and sum(len(t) == 1 for t in toks) / len(toks) >= 0.8:
            words = re.split(r" {2,}", line.strip())
            lines.append(" ".join(w.replace(" ", "") for w in words))
        else:
            lines.append(line)
    return "\n".join(lines)


def _printable(s: str) -> bool:
    return bool(s) and sum(c.isprintable() or c in "\n\t\r" for c in s) / len(s) > 0.95


def decode_strict(seg: str) -> str:
    """Decode a segment that is entirely Base64, hex or Morse (format-checked, never guessed)."""
    s = seg.strip()
    if not s:
        return seg
    if _B64.match(s) and len(s) % 4 == 0:
        try:
            dec = base64.b64decode(s, validate=True).decode("utf-8")
            if _printable(dec):
                return dec
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
    if _HEX.match(s):
        try:
            dec = bytes.fromhex(s).decode("utf-8")
            if _printable(dec):
                return dec
        except (ValueError, UnicodeDecodeError):
            pass
    if _MORSE.match(s) and sum(c in ".-" for c in s) >= 4:
        dec = from_morse(s)
        if dec.strip():
            return dec
    return seg


def decode_heuristic(seg: str, min_score: float = 0.28, margin: float = 0.15, min_pairs: int = 10) -> str:
    """Undo ROT13 or character reversal if the result is clearly more English-like."""
    s = seg.strip()
    base, pairs = english_score(s)
    if pairs < min_pairs:
        return seg
    best, best_score = None, max(min_score, base + margin)
    for cand in (rot13(s), s[::-1]):
        score, _ = english_score(cand)
        if score > best_score:
            best, best_score = cand, score
    return best if best is not None else seg


def deleet(text: str) -> str:
    toks = text.split()
    if not toks:
        return text
    mixed = sum(bool(re.search(r"[A-Za-z]", t)) and bool(re.search(r"[0-9@$]", t)) for t in toks)
    return text.translate(_LEET_BACK) if mixed / len(toks) >= 0.2 else text


def normalize_case(text: str) -> str:
    """lOwEr-CaSe text with erratic capitalisation (but leave normal and ALL-CAPS text alone)."""
    letters = [c for c in text if c.isascii() and c.isalpha()]
    if len(letters) < 10:
        return text
    upper = sum(c.isupper() for c in letters) / len(letters)
    return text.lower() if 0.3 <= upper <= 0.7 else text


def canonicalize(text: str) -> str:
    t = decode_tags(text)
    t = _INVISIBLE.sub("", t)
    t = unicodedata.normalize("NFKC", t)
    t = strip_latin_marks(t)
    t = map_confusables(t)
    t = collapse_spacing(t)
    t = "\n\n".join(decode_strict(seg) for seg in t.split("\n\n"))
    t = deleet(t)
    t = "\n\n".join(decode_heuristic(seg) for seg in t.split("\n\n"))
    t = normalize_case(t)
    return t.strip()
