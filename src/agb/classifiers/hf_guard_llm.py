"""Generative guard models run through Hugging Face transformers (e.g. Qwen3Guard-Gen).

The model's own chat template renders the moderation prompt. We append the fixed
start of the verdict (e.g. ``"Safety:"``) and read the next-token distribution over
the first tokens of the verdict labels, which yields a calibrated-ish probability per
label from a single forward pass (no sampling).

The long policy text that precedes the user content is identical for every input, so
its KV cache is computed once and re-used (verified numerically at start-up; falls
back to full forward passes if the check fails).
"""

from __future__ import annotations

import os

from .base import Classifier

_PLACEHOLDER = "\u2063AGB_CONTENT_PLACEHOLDER\u2063"


class HFGuardLLM(Classifier):
    def __init__(
        self,
        model_id: str,
        labels: dict[str, str],
        variants: dict[str, list[str]],
        assistant_prefix: str = "",
        name: str | None = None,
        revision: str | None = None,
        use_prefix_cache: bool = True,
        max_content_tokens: int = 4096,
        log=print,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self._torch = torch
        self.log = log
        self.name = name or model_id
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.variants = tuple(variants)
        self.variant_labels = variants
        self.max_content_tokens = max_content_tokens

        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": _PLACEHOLDER}], tokenize=False, add_generation_prompt=True
        )
        if _PLACEHOLDER not in rendered:
            raise ValueError("chat template dropped the user content")
        pre, post = rendered.split(_PLACEHOLDER, 1)
        cut = pre.rfind("\n") + 1  # split the cacheable prefix on a newline (a safe token boundary)
        self.cache_text, self.head, self.tail = pre[:cut], pre[cut:], post + assistant_prefix
        self.template_tail = self.tail

        self.label_names = list(labels)
        first = [self.tok.encode(cont, add_special_tokens=False)[0] for cont in labels.values()]
        if len(set(first)) != len(first):
            raise ValueError(f"label continuations share a first token: {labels}")
        self.label_ids = first

        self.prefix_ids = self._ids(self.cache_text)
        self.prefix_len = int(self.prefix_ids.shape[1])
        self.cache = None
        self.cache_ok = False
        if use_prefix_cache and self.prefix_len > 0:
            self._init_cache()

    # ------------------------------------------------------------------ helpers
    def _ids(self, text: str):
        return self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids

    def _forward_last(self, ids, cache=None):
        torch = self._torch
        with torch.inference_mode():
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=cache is not None, logits_to_keep=1)
        return out.logits[0, -1].float()

    def _init_cache(self) -> None:
        torch = self._torch
        from transformers import DynamicCache

        probe = "How do I bake sourdough bread at home?"
        full = self._ids(self.cache_text + self.head + probe + self.tail)
        split = torch.cat([self.prefix_ids, self._ids(self.head + probe + self.tail)], dim=1)
        if full.shape != split.shape or not torch.equal(full, split):
            self.log(f"[{self.name}] prefix tokenization differs; prefix cache disabled")
            return
        cache = DynamicCache()
        with torch.inference_mode():
            self.model(input_ids=self.prefix_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        ref = self._forward_last(full)
        got = self._forward_last(self._ids(self.head + probe + self.tail), cache)
        cache.crop(self.prefix_len)
        err = float((ref - got).abs().max())
        if err > 1e-2 * max(1.0, float(ref.abs().max())):
            self.log(f"[{self.name}] cached logits differ (max abs err {err:.3g}); prefix cache disabled")
            return
        self.cache, self.cache_ok = cache, True
        self.log(f"[{self.name}] prefix cache enabled: {self.prefix_len} tokens (max abs err {err:.2g})")

    def _content_ids(self, text: str):
        ids = self._ids(self.head + text + self.tail)
        if ids.shape[1] <= self.max_content_tokens:
            return ids, False
        t_ids = self.tok.encode(text, add_special_tokens=False)[: self.max_content_tokens]
        text = self.tok.decode(t_ids)
        return self._ids(self.head + text + self.tail), True

    # ------------------------------------------------------------------ API
    def score_texts(self, texts: list[str]) -> list[dict]:
        torch = self._torch
        results = []
        for text in texts:
            ids, truncated = self._content_ids(text)
            if self.cache_ok:
                logits = self._forward_last(ids, self.cache)
                self.cache.crop(self.prefix_len)
            else:
                logits = self._forward_last(torch.cat([self.prefix_ids, ids], dim=1))
            sub = logits[self.label_ids]
            probs = torch.softmax(sub, dim=-1).tolist()
            mass = float(torch.exp(torch.logsumexp(sub, 0) - torch.logsumexp(logits, 0)))
            pd = dict(zip(self.label_names, probs))
            scores = {v: float(sum(pd[lab] for lab in labs)) for v, labs in self.variant_labels.items()}
            verdict = max(pd, key=pd.get)
            results.append({"scores": scores, "out": f"{verdict}|mass={mass:.2f}{'|trunc' if truncated else ''}"})
        return results

    def generate_probe(self, text: str, max_new_tokens: int = 24) -> str:
        """Greedy continuation after the verdict prefix (start-up sanity check of the output format)."""
        torch = self._torch
        ids = self._ids(self.cache_text + self.head + text + self.tail)
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    def info(self) -> dict:
        return {
            "backend": "hf_guard_llm",
            "model_id": self.model_id,
            "revision": self.revision,
            "labels": dict(zip(self.label_names, self.label_ids)),
            "prefix_cache": self.cache_ok,
            "prefix_tokens": self.prefix_len,
            "template_tail": self.template_tail[-120:],
        }
