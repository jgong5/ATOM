# SPDX-License-Identifier: MIT
"""A simulated run charges the encode ATOM's segment tokenizer makes of a long
prompt: it encodes the segments no earlier prompt held through the backend's
``encode_batch``, past the served tokenizer's ``encode``.

As in `test_served_tokenizer`, a run file names the machine spec of
`test_memory_readings`, whose tokenizer entry applies to ``Qwen3ForCausalLM`` on
the fast backend, and the calls are made on the clock owner's thread.
"""

import json
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from test_memory_readings import DOCUMENT, TOKENIZER  # noqa: E402
from tokenizers import (  # noqa: E402
    AddedToken,
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)

from atom.compass import run as compass_run  # noqa: E402
from atom.compass.clock import LpId, single_engine_table  # noqa: E402
from atom.model_engine import segment_tokenizer as seg_mod  # noqa: E402
from atom.model_engine.segment_tokenizer import SegmentTokenizer  # noqa: E402
from atom.utils import clock  # noqa: E402
from atom.utils.clock import LPRuntime  # noqa: E402

SPECIALS = ["<|begin|>", "<|User|>", "<|Assistant|>"]
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "12345", "hello,", "world!"]
CONFIG = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]))


def _text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _plain(tok, prompt):
    """The served tokenizer's own ``encode``, past any wrapper on the instance."""
    return transformers.PreTrainedTokenizerFast.encode(tok, prompt)


@pytest.fixture
def served(monkeypatch):
    """A fast tokenizer and its segment tokenizer, past the prompts it checks
    against the plain encoder, taking prompts of any length."""
    monkeypatch.setattr(seg_mod, "_MIN_CACHED_CHARS", 0)
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
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend)
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    assert seg is not None and seg._prefix_ok
    seg._verified = seg_mod._VERIFY_FIRST
    return tok, seg


class _GrantAll:
    def send(self, msg):
        self.t = msg[1]

    def recv(self):
        return self.t, {}


@pytest.fixture
def rt(monkeypatch, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"bound_s": 600.0, "machine": DOCUMENT}))
    monkeypatch.setenv(compass_run.ENV, str(path))
    rt = LPRuntime(
        LpId("frontend"),
        single_engine_table(admission_path="serving", ipc_s=0.001, stream_s=0.002),
        _GrantAll(),
    )
    clock.install(rt)
    rt.start_run()
    yield rt
    clock.install(None)


def test_each_long_prompt_costs_one_encode_of_the_ids_it_encoded(served, rt):
    tok, seg = served
    compass_run.tokenizer(tok, CONFIG, seg)
    rate = TOKENIZER["encode_tokens_per_s"] * TOKENIZER["derate"]
    rng = random.Random(1)
    # Longer than the characters that bucket a stored prompt, so a later turn
    # finds the one before it.
    prompt = "<|begin|>" + _text(rng, 3000)
    for turn in range(4):
        prompt += "<|User|>" + _text(rng, 40) + "<|Assistant|>"
        t, before = rt.now, seg._stats[1]
        got = seg.encode_with_prefix(prompt, 4).token_ids
        cost, encoded = rt.now - t, seg._stats[1] - before
        print(f"\n  turn {turn}: {len(got)} ids, {encoded} encoded, {cost!r} s")
        assert got == _plain(tok, prompt)
        assert cost == pytest.approx(TOKENIZER["encode_fixed_s"] + encoded / rate)
        if turn:
            # The earlier turn's ids are reused, and cost nothing.
            assert 0 < encoded < len(got) // 10


def test_with_no_run_file_the_segment_tokenizer_is_left_as_it_is(served, monkeypatch):
    tok, seg = served
    monkeypatch.delenv(compass_run.ENV, raising=False)
    backend = seg._split.backend
    compass_run.tokenizer(tok, CONFIG, seg)
    assert seg._split.backend is backend and vars(tok).get("encode") is None
    prompt = "<|begin|>" + _text(random.Random(2), 3000) + "<|User|>"
    assert seg.encode_with_prefix(prompt, 4).token_ids == _plain(tok, prompt)
