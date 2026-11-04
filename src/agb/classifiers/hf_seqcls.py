"""Hugging Face sequence classifiers: Prompt Guard v1/v2, ProtectAI DeBERTa, ...

Long inputs are split into overlapping windows of the model's maximum length and the
text is scored by its most suspicious window (Meta's recommended way of applying
Prompt Guard to long inputs; naive truncation would let a payload hide past token 512).
"""

from __future__ import annotations

import os

from .base import Classifier

BINARY_POSITIVE_ALIASES = {"MALICIOUS", "LABEL_1", "INJECTION", "JAILBREAK", "UNSAFE", "1"}


def resolve_label_indices(id2label: dict[int, str], wanted: list[str]) -> list[int]:
    """Map label names from the config to output indices (case-insensitive).

    For binary models whose config only has generic names, index 1 is the positive class.
    """
    names = {i: str(n).upper() for i, n in id2label.items()}
    want = {w.upper() for w in wanted}
    idx = sorted(i for i, n in names.items() if n in want)
    if not idx and len(names) == 2 and want & BINARY_POSITIVE_ALIASES:
        idx = [1]
    if not idx:
        raise ValueError(f"none of {wanted} found in id2label={id2label}")
    return idx


class HFSequenceClassifier(Classifier):
    def __init__(
        self,
        model_id: str,
        variants: dict[str, list[str]],
        name: str | None = None,
        revision: str | None = None,
        batch_size: int = 16,
        overlap: int = 64,
        temperature: float = 1.0,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # silence "sequence longer than max length" warnings
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self._torch = torch
        self.name = name or model_id
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.variants = tuple(variants)
        self.variant_idx = {v: resolve_label_indices(self.id2label, labels) for v, labels in variants.items()}
        max_len = min(512, int(getattr(self.tok, "model_max_length", 512) or 512))
        self.window = max_len - self.tok.num_special_tokens_to_add(pair=False)
        self.overlap = min(overlap, self.window // 4)
        self.batch_size = batch_size
        self.temperature = temperature
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def _windows(self, ids: list[int]) -> list[list[int]]:
        if len(ids) <= self.window:
            return [ids]
        step = self.window - self.overlap
        return [ids[i : i + self.window] for i in range(0, len(ids) - self.overlap, step)]

    def score_texts(self, texts: list[str]) -> list[dict]:
        torch = self._torch
        enc = self.tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
        flat: list[tuple[int, list[int]]] = []
        for ti, ids in enumerate(enc):
            for w in self._windows(ids):
                flat.append((ti, self.tok.build_inputs_with_special_tokens(w)))
        order = sorted(range(len(flat)), key=lambda k: len(flat[k][1]))
        probs: list = [None] * len(flat)
        for s in range(0, len(order), self.batch_size):
            chunk = order[s : s + self.batch_size]
            batch = self.tok.pad({"input_ids": [flat[k][1] for k in chunk]}, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**batch).logits.float() / self.temperature
            p = torch.softmax(logits, dim=-1).numpy()
            for k, row in zip(chunk, p):
                probs[k] = row
        results = [{"scores": {v: 0.0 for v in self.variants}, "out": ""} for _ in texts]
        best = [-1.0] * len(texts)
        for (ti, _), row in zip(flat, probs):
            for v, idx in self.variant_idx.items():
                s = float(row[idx].sum())
                if s > results[ti]["scores"][v]:
                    results[ti]["scores"][v] = s
            top = float(row.max())
            if top > best[ti] or not results[ti]["out"]:
                best[ti] = top
                results[ti]["out"] = self.id2label[int(row.argmax())]
        return results

    def info(self) -> dict:
        return {
            "backend": "hf_seqcls",
            "model_id": self.model_id,
            "revision": self.revision,
            "id2label": self.id2label,
            "variant_indices": self.variant_idx,
            "window_tokens": self.window,
        }
