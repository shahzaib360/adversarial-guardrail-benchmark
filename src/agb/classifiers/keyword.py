"""Keyword/regex block-list baselines -- what a naive filter achieves."""

from __future__ import annotations

import re

from .base import Classifier

HARM_TERMS = [
    r"bombs?", r"explosives?", r"detonat\w*", r"weapons?", r"firearms?", r"guns?", r"ammunition",
    r"kill(s|ing|ed)?", r"murder\w*", r"assassinat\w*", r"poison\w*", r"toxins?", r"nerve agents?",
    r"sarin", r"ricin", r"anthrax", r"pathogens?", r"bio-?weapons?", r"chemical weapons?",
    r"meth(amphetamine)?", r"cocaine", r"heroin", r"fentanyl", r"drugs?", r"narcotics?", r"hack\w*",
    r"malware", r"ransomware", r"keyloggers?", r"exploit\w*", r"phishing", r"ddos", r"botnets?",
    r"steal\w*", r"theft", r"fraud\w*", r"scam\w*", r"launder\w*", r"counterfeit\w*", r"smuggl\w*",
    r"traffick\w*", r"suicide", r"self-harm", r"terror\w*", r"extremis\w*", r"racis\w*", r"slurs?",
    r"harass\w*", r"stalk\w*", r"dox\w*", r"blackmail\w*", r"extort\w*", r"abuse", r"tortur\w*",
    r"violen\w*", r"illegal(ly)?", r"disinformation", r"misinformation", r"propaganda", r"hate",
    r"nazi\w*", r"genocide",
]  # fmt: skip

INJECTION_TERMS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier|preceding) (instructions?|prompts?|rules|text)",
    r"disregard", r"forget (all |your |the )?(previous |prior )?(instructions?|rules)", r"you are now",
    r"from now on", r"developer mode", r"\bDAN\b", r"do anything now", r"jailbr\w*", r"unfiltered",
    r"uncensored", r"no (restrictions|limits|limitations|rules|filters|censorship)",
    r"without (any )?(restrictions|limitations|censorship|filters)", r"system prompt",
    r"pretend (to be|you are)", r"act as", r"role-?play", r"stay in character", r"access granted",
    r"override", r"bypass", r"unrestricted", r"opposite mode", r"evil", r"your (instructions|rules)",
]  # fmt: skip

LEXICONS = {"harm": HARM_TERMS, "injection": INJECTION_TERMS}


class KeywordClassifier(Classifier):
    continuous = False

    def __init__(self, lexicon: str = "harm", name: str = "keyword"):
        self.name = name
        self.lexicon = lexicon
        self.pattern = re.compile(r"\b(?:" + "|".join(LEXICONS[lexicon]) + r")\b", re.IGNORECASE)

    def score_texts(self, texts: list[str]) -> list[dict]:
        out = []
        for t in texts:
            m = self.pattern.search(t)
            out.append({"scores": {"default": 1.0 if m else 0.0}, "out": "hit" if m else ""})
        return out

    def info(self) -> dict:
        return {"backend": "keyword", "lexicon": self.lexicon, "n_terms": len(LEXICONS[self.lexicon])}
