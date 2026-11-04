"""Guard models served by a local Ollama server (e.g. ``llama-guard3:1b`` / ``llama-guard3:8b``).

Ollama's library ships Meta's Llama Guard 3 weights (quantized GGUF) with the official
prompt template, and llama.cpp makes CPU inference practical. The verdict is the first
line of the reply (``safe`` / ``unsafe`` + hazard categories S1..S14). When the server
returns token log-probabilities we turn them into P(unsafe); otherwise the score is the
0/1 verdict.
"""

from __future__ import annotations

import math
import os
import re
import time

import requests

from .base import Classifier

_CATS = re.compile(r"\bS\d{1,2}\b")


def _norm_tok(tok: str) -> str:
    return tok.strip().lower()


def prob_unsafe_from_logprobs(logprobs, safe: str = "safe", unsafe: str = "unsafe") -> float | None:
    """P(unsafe) at the first position whose token starts the verdict, renormalised over safe/unsafe."""
    if not logprobs:
        return None
    for entry in logprobs:
        tok = _norm_tok(entry.get("token", ""))
        if not tok:
            continue  # leading whitespace / newline tokens
        cands = entry.get("top_logprobs") or [entry]
        p_safe = p_unsafe = 0.0
        for c in cands:
            t = _norm_tok(c.get("token", ""))
            p = math.exp(float(c.get("logprob", -1e9)))
            if t and unsafe.startswith(t) and t != safe and len(t) >= 2:  # "unsafe", "un", "uns"...
                p_unsafe += p
            elif t and (t == safe or t.startswith(safe)):
                p_safe += p
        if p_safe + p_unsafe > 0:
            return p_unsafe / (p_safe + p_unsafe)
        return None
    return None


class OllamaGuard(Classifier):
    def __init__(
        self,
        model: str,
        name: str | None = None,
        host: str | None = None,
        num_ctx: int = 4096,
        timeout: int = 900,
        top_logprobs: int = 10,
    ):
        self.model = model
        self.name = name or model
        self.host = (host or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.top_logprobs = top_logprobs
        self.session = requests.Session()
        self.logprobs_supported: bool | None = None
        self.server_version = self._version()

    def _version(self) -> str:
        try:
            return self.session.get(f"{self.host}/api/version", timeout=30).json().get("version", "?")
        except Exception:  # noqa: BLE001
            return "?"

    def _chat(self, text: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": 16, "num_ctx": self.num_ctx},
        }
        if self.logprobs_supported is not False:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs
        for attempt in range(5):
            try:
                r = self.session.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                if r.status_code == 400 and "logprobs" in payload:
                    self.logprobs_supported = False
                    payload.pop("logprobs"), payload.pop("top_logprobs")
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("unreachable")

    def score_texts(self, texts: list[str]) -> list[dict]:
        out = []
        for text in texts:
            js = self._chat(text)
            content = (js.get("message") or {}).get("content", "") or ""
            first = next((ln.strip().lower() for ln in content.splitlines() if ln.strip()), "")
            verdict_unsafe = first.startswith("unsafe")
            p = prob_unsafe_from_logprobs(js.get("logprobs"))
            if self.logprobs_supported is None:
                self.logprobs_supported = p is not None
            # Guard against rare disagreement between the logprob read-out and the decoded verdict.
            score = p if p is not None else (1.0 if verdict_unsafe else 0.0)
            cats = ",".join(_CATS.findall(content))
            out.append(
                {
                    "scores": {"default": float(score)},
                    "out": ("unsafe" if verdict_unsafe else "safe") + (f":{cats}" if cats else ""),
                }
            )
        return out

    @property
    def continuous(self) -> bool:  # type: ignore[override]
        return bool(self.logprobs_supported)

    def _details(self) -> dict:
        try:
            js = self.session.post(f"{self.host}/api/show", json={"model": self.model}, timeout=60).json()
            d = js.get("details") or {}
            return {k: d.get(k) for k in ("family", "parameter_size", "quantization_level", "format")}
        except Exception:  # noqa: BLE001
            return {}

    def info(self) -> dict:
        return {
            "backend": "ollama",
            "model": self.model,
            "ollama_version": self.server_version,
            "logprobs": self.logprobs_supported,
            "num_ctx": self.num_ctx,
            **self._details(),
        }
