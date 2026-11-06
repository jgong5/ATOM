# SPDX-License-Identifier: MIT
"""A prompt's ids go from the segment tokenizer to its `Sequence` as the
``array("i")`` the `Sequence` keeps them in, not as a list.

`Sequence` copies an array of its own typecode as one buffer, and a list one
boxed id at a time; a list in between also costs a `tolist` over every id. The
processor here is ATOM's own `InputOutputProcessor` on a small trained
tokenizer, with the segment tokenizer `__init__` builds, as the API server's
pool job runs it.
"""

import array
import random
from types import SimpleNamespace

import pytest
import transformers
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers

from atom.model_engine import llm_engine
from atom.model_engine.llm_engine import InputOutputProcessor
from atom.model_engine.segment_tokenizer import _MIN_CACHED_CHARS, _VERIFY_FIRST
from atom.model_engine.sequence import Sequence, new_token_ids
from atom.sampling_params import SamplingParams

SPECIALS = ["<|begin|>", "<|User|>", "<|Assistant|>"]
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


@pytest.fixture(scope="module")
def tok():
    rng = random.Random(0)
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        [_text(rng, 40) for _ in range(200)],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    backend.add_special_tokens([AddedToken(t, special=True) for t in SPECIALS])
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend)


def _turns(n):
    """One conversation's prompts, each past `_MIN_CACHED_CHARS` and each
    extending the one before, so later turns reuse the stored prompt."""
    rng = random.Random(1)
    history = "<|begin|>" + _text(rng, 50)
    while len(history) < _MIN_CACHED_CHARS:
        history += "<|User|>" + _text(rng, 200) + "<|Assistant|>" + _text(rng, 100)
    prompts = []
    for _ in range(n):
        history += "<|User|>" + _text(rng, 30)
        prompts.append(history + "<|Assistant|>")
        history += "<|Assistant|>" + _text(rng, 20)
    return prompts


@pytest.fixture
def handed(monkeypatch):
    """The `token_ids` each `Sequence` the processor builds is given."""
    seen = []

    class Recording(Sequence):
        def __init__(self, token_ids, *args, **kwargs):
            seen.append(token_ids)
            super().__init__(token_ids, *args, **kwargs)

    monkeypatch.setattr(llm_engine, "Sequence", Recording)
    return seen


@pytest.mark.parametrize("prefix_hash", ["1", "0"], ids=["with-hashes", "ids-only"])
def test_a_prompts_ids_reach_its_sequences_as_an_int32_array(
    tok, handed, monkeypatch, prefix_hash
):
    monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", prefix_hash)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3"),
        speculative_config=None,
        enable_prefix_caching=True,
        decode_context_parallel_size=1,
    )
    proc = InputOutputProcessor(config, tok, 16)
    assert proc.segment_tokenizer is not None
    # A short prompt, then turns checked against the plain encoder and past it.
    prompts = ["<|begin|>" + _text(random.Random(2), 40)]
    prompts += _turns(_VERIFY_FIRST + 2)
    for prompt in prompts:
        seqs = proc.preprocess_fanout(prompt, SamplingParams(n=2))
        ids = handed[-1]
        assert type(ids) is array.array and ids.typecode == "i", type(ids)
        plain = new_token_ids(tok.encode(prompt))
        for seq in seqs:
            assert seq.token_ids == plain
            assert seq.token_ids is not ids
        seqs[0].token_ids[0] = -7
        assert seqs[1].token_ids == plain
    assert proc.segment_tokenizer.enabled
    assert proc.segment_tokenizer._stats[4] == len(prompts) - 2  # prefix hits
