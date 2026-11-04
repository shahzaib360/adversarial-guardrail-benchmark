# Adversarial Guardrail Benchmark

Stress-tests open guardrail classifiers — **Llama Guard 3**, **Prompt Guard (v1 and 2)**, Qwen3Guard and
ProtectAI's injection detector — with encoding, Unicode, structural, framing, paraphrase and translation
attacks, and reports **miss rates alongside false-positive rates** measured on benign inputs that receive
exactly the same transformations.

Full documentation and results are being added; see `configs/benchmark.yaml` for the experiment definition.
