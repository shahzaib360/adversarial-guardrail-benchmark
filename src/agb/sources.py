"""Pinned, GitHub-hosted dataset sources.

Every file is fetched at runtime from the original authors' repository at a fixed
commit and verified against a SHA-256 checksum. Nothing is redistributed in this
repository.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import time
from dataclasses import dataclass

import requests


def cache_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("AGB_CACHE", pathlib.Path.home() / ".cache" / "agb"))


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    commit: str
    path: str
    sha256: str
    license: str
    citation: str

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{self.path}"

    @property
    def homepage(self) -> str:
        return f"https://github.com/{self.repo}"


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            name="harmbench",
            repo="centerforaisafety/HarmBench",
            commit="8e1604d1171fe8a48d8febecd22f600e462bdcdd",
            path="data/behavior_datasets/harmbench_behaviors_text_all.csv",
            sha256="8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc",
            license="MIT",
            citation="Mazeika et al., HarmBench (2024)",
        ),
        Source(
            name="xstest",
            repo="paul-rottger/xstest",
            commit="d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
            path="xstest_prompts.csv",
            sha256="11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
            license="CC-BY-4.0",
            citation="Röttger et al., XSTest (NAACL 2024)",
        ),
        Source(
            name="jailbreakhub_jailbreak",
            repo="verazuo/jailbreak_llms",
            commit="4f4031bf8be187f4478c7f94f42b08714722c12e",
            path="data/prompts/jailbreak_prompts_2023_12_25.csv",
            sha256="accf97463de96c33c48453dfee9600191ff6fd5a0c43f3f11d5ebf98d1a5de55",
            license="MIT",
            citation="Shen et al., 'Do Anything Now' (CCS 2024)",
        ),
        Source(
            name="jailbreakhub_regular",
            repo="verazuo/jailbreak_llms",
            commit="4f4031bf8be187f4478c7f94f42b08714722c12e",
            path="data/prompts/regular_prompts_2023_12_25.csv",
            sha256="fb82e3f88fbd5d9c6edf8927b138510f339b85012cab24d41d13e1665a9c5819",
            license="MIT",
            citation="Shen et al., 'Do Anything Now' (CCS 2024)",
        ),
        Source(
            name="alpaca",
            repo="tatsu-lab/stanford_alpaca",
            commit="761dc5bfbdeeffa89b8bff5d038781a4055f796a",
            path="alpaca_data.json",
            sha256="2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419",
            license="CC-BY-NC-4.0",
            citation="Taori et al., Stanford Alpaca (2023)",
        ),
        Source(
            name="tensortrust_hijacking",
            repo="HumanCompatibleAI/tensor-trust-data",
            commit="747a75e096761ebc01bd3970158827326b4add23",
            path="benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl",
            sha256="2ad3c2343bbf43cb1924f3ec5d6f830864ce3c879e9832b01d93a991e7e2e680",
            license="no license file (research benchmark; fetched at runtime, not redistributed)",
            citation="Toyer et al., Tensor Trust (ICLR 2024)",
        ),
        Source(
            name="tensortrust_extraction",
            repo="HumanCompatibleAI/tensor-trust-data",
            commit="747a75e096761ebc01bd3970158827326b4add23",
            path="benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl",
            sha256="4090f9f57b6df69cc32d7669ac991e3fd53e6e2bb616a5e6f45dac79c183ee4c",
            license="no license file (research benchmark; fetched at runtime, not redistributed)",
            citation="Toyer et al., Tensor Trust (ICLR 2024)",
        ),
    ]
}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, retries: int = 4, timeout: int = 120) -> pathlib.Path:
    """Download (or reuse a cached copy of) a pinned source file and verify its checksum."""
    src = SOURCES[name]
    dest = cache_dir() / src.name / pathlib.PurePosixPath(src.path).name
    if dest.exists() and _sha256(dest) == src.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(src.url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            digest = _sha256(tmp)
            if digest != src.sha256:
                raise ValueError(f"checksum mismatch for {src.name}: got {digest}, expected {src.sha256}")
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001 - retry on any network/checksum error
            last_err = e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"could not fetch {src.url}: {last_err}")
