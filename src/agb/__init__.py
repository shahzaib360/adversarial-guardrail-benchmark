"""Adversarial Guardrail Benchmark (agb).

Stress-tests open guardrail classifiers (Llama Guard, Prompt Guard, ...) with
encoding, Unicode, structural, framing, paraphrase and translation attacks, and
reports miss rates together with false-positive rates on benign inputs that
receive exactly the same transformations.
"""

__version__ = "0.1.0"
