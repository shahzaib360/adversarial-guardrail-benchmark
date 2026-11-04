"""Classifier factory: builds a classifier from its entry in ``configs/benchmark.yaml``."""

from __future__ import annotations

from .base import Classifier


def build_classifier(name: str, cfg: dict, revision: str | None = None, log=print) -> Classifier:
    backend = cfg["backend"]
    if backend == "keyword":
        from .keyword import KeywordClassifier

        return KeywordClassifier(lexicon=cfg.get("lexicon", "harm"), name=name)
    if backend == "hf_seqcls":
        from .hf_seqcls import HFSequenceClassifier

        return HFSequenceClassifier(
            cfg["model_id"],
            variants=cfg["variants"],
            name=name,
            revision=revision,
            batch_size=int(cfg.get("batch_size", 16)),
        )
    if backend == "hf_guard_llm":
        from .hf_guard_llm import HFGuardLLM

        return HFGuardLLM(
            cfg["model_id"],
            labels=cfg["labels"],
            variants=cfg["variants"],
            assistant_prefix=cfg.get("assistant_prefix", ""),
            name=name,
            revision=revision,
            max_content_tokens=int(cfg.get("max_content_tokens", 4096)),
            log=log,
        )
    if backend == "ollama":
        from .ollama_guard import OllamaGuard

        return OllamaGuard(cfg["model_id"], name=name, num_ctx=int(cfg.get("num_ctx", 4096)))
    raise ValueError(f"unknown backend {backend!r} for model {name!r}")


def variant_names(cfg: dict) -> list[str]:
    return list(cfg.get("variants", {"default": None}))


__all__ = ["Classifier", "build_classifier", "variant_names"]
