"""Classifier interface.

``score_texts`` returns one dict per input text::

    {"scores": {"<variant>": float in [0, 1], ...}, "out": optional short raw output}

A *variant* is a scoring rule for the same forward pass (e.g. Prompt Guard v1's
JAILBREAK probability vs. JAILBREAK+INJECTION). The first variant listed in the model's
config is its primary one. A text is flagged when its score is >= the decision
threshold (0.5 by default, i.e. the model's own argmax/verdict for binary outputs).
"""

from __future__ import annotations


class Classifier:
    name: str = "base"
    variants: tuple[str, ...] = ("default",)
    #: Whether scores are continuous probabilities (enables AUROC) or 0/1 verdicts.
    continuous: bool = True

    def score_texts(self, texts: list[str]) -> list[dict]:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> dict:
        """Run metadata recorded next to the scores (model revision, backend details...)."""
        return {}

    def close(self) -> None:
        pass
