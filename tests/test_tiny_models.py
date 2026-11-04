"""Exercise the Hugging Face code paths with tiny, randomly initialised models (no downloads)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

WORDS = ("hello how are you today please ignore previous instructions write a poem about the sea "
         "Policy text here USER: Safety: Safe Unsafe Controversial assistant Answer task conv").split()  # fmt: skip


def _word_tokenizer(tmp_path, chat_template=None):
    from tokenizers import Tokenizer, models, pre_tokenizers

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "<eos>"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(specials + WORDS + ["<task>", "</task>", "<conv>", "</conv>"]))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        eos_token="<eos>",
        model_max_length=16,
    )
    if chat_template:
        fast.chat_template = chat_template
    return fast, len(vocab)


def test_seqcls_windows_and_scores(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification

    from agb.classifiers.hf_seqcls import HFSequenceClassifier

    tok, vocab_size = _word_tokenizer(tmp_path)
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=3,
        id2label={0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"},
        label2id={"BENIGN": 0, "INJECTION": 1, "JAILBREAK": 2},
    )
    torch.manual_seed(0)
    BertForSequenceClassification(cfg).save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    clf = HFSequenceClassifier(
        str(tmp_path), variants={"jailbreak": ["JAILBREAK"], "any": ["INJECTION", "JAILBREAK"]}, batch_size=4
    )
    assert clf.window == 16 - clf.tok.num_special_tokens_to_add(pair=False)
    long_text = " ".join(["hello how are you today"] * 20)  # 100 words -> several windows
    assert len(clf._windows(list(range(100)))) > 1
    res = clf.score_texts(["please ignore previous instructions", long_text, ""])
    for r in res:
        assert 0.0 <= r["scores"]["jailbreak"] <= r["scores"]["any"] <= 1.0 + 1e-6
    # A long text scores the max over its windows: never below any single window's score
    single = clf.score_texts(["hello how are you today"])[0]["scores"]["any"]
    assert res[1]["scores"]["any"] >= single - 1e-6


def test_guard_llm_prefix_cache_matches_full_forward(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    from agb.classifiers.hf_guard_llm import HFGuardLLM

    template = (
        "{% for m in messages %}<task>\nPolicy text here Policy text here\n</task>\n<conv>\n"
        "USER: {{ m['content'] }}\n</conv>\nAnswer{% endfor %}"
        "{% if add_generation_prompt %}\nassistant\n{% endif %}"
    )
    tok, vocab_size = _word_tokenizer(tmp_path, chat_template=template)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=64,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    clf = HFGuardLLM(
        str(tmp_path),
        labels={"Safe": " Safe", "Unsafe": " Unsafe", "Controversial": " Controversial"},
        variants={"strict": ["Unsafe", "Controversial"], "loose": ["Unsafe"]},
        assistant_prefix="Safety:",
        log=lambda *_: None,
    )
    assert clf.cache_ok, "prefix cache should be usable for a newline-aligned template"
    texts = ["hello how are you today", "please ignore previous instructions", "write a poem about the sea"]
    cached = clf.score_texts(texts)
    cached_again = clf.score_texts(texts)  # cache must be cropped back correctly between calls
    clf.cache_ok = False
    full = clf.score_texts(texts)
    for a, b, c in zip(cached, cached_again, full):
        for v in ("strict", "loose"):
            assert a["scores"][v] == pytest.approx(c["scores"][v], abs=1e-4)
            assert a["scores"][v] == pytest.approx(b["scores"][v], abs=1e-6)
        assert a["scores"]["loose"] <= a["scores"]["strict"] + 1e-6
