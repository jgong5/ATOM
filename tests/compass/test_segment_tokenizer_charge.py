# SPDX-License-Identifier: MIT
"""A simulated run charges the encodes ATOM's segment tokenizer makes.

`InputOutputProcessor` tokenizes a prompt of `_MIN_CACHED_CHARS` characters or
more through its `SegmentTokenizer`, which encodes the segments it has not
cached in one ``encode_batch`` on the fast backend, past the served tokenizer's
``encode``. Here the processor is built on a small trained tokenizer before
`compass_run.tokenizer` wraps it, as the API server builds its engine first, and
requests are preprocessed on the clock owner's thread, so each charge advances
the LP clock. The run file names the machine spec of `test_memory_readings`.
"""

import json
import random
from types import SimpleNamespace

import pytest
import transformers
from test_memory_readings import DOCUMENT, TOKENIZER
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers

from atom.compass import run as compass_run
from atom.compass.clock import LpId, single_engine_table
from atom.model_engine.llm_engine import InputOutputProcessor
from atom.model_engine.segment_tokenizer import _MIN_CACHED_CHARS, _VERIFY_FIRST
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.clock import LPRuntime

SPECIALS = ["<|begin|>", "<|User|>", "<|Assistant|>"]
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _law(ids):
    rate = TOKENIZER["encode_tokens_per_s"] * TOKENIZER["derate"]
    return TOKENIZER["encode_fixed_s"] + ids / rate


def _text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _tokenizer():
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


def _prompt(seed):
    """A conversation of `_MIN_CACHED_CHARS` characters, its own text."""
    rng = random.Random(seed)
    prompt = "<|begin|>" + _text(rng, 50)
    while len(prompt) < _MIN_CACHED_CHARS:
        prompt += "<|User|>" + _text(rng, 200) + "<|Assistant|>" + _text(rng, 100)
    return prompt[:_MIN_CACHED_CHARS]


def _config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3", architectures=["Qwen3ForCausalLM"]
        ),
        speculative_config=None,
        enable_prefix_caching=True,
        decode_context_parallel_size=1,
    )


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


@pytest.fixture
def proc(rt):
    tok, config = _tokenizer(), _config()
    proc = InputOutputProcessor(config, tok, 16)
    assert proc.segment_tokenizer is not None
    compass_run.tokenizer(tok, config)
    return proc


def _preprocess(rt, proc, prompt):
    """LP seconds the request was charged, and the ids its segment tokenizer
    counts as encoded rather than reused."""
    stats = proc.segment_tokenizer._stats
    t, encoded = rt.now, stats[1]
    proc.preprocess(prompt, SamplingParams())
    return rt.now - t, stats[1] - encoded


def test_a_long_prompt_after_the_verified_ones_is_charged_the_ids_it_encoded(rt, proc):
    for seed in range(_VERIFY_FIRST):
        _preprocess(rt, proc, _prompt(seed))
    prompt = _prompt(_VERIFY_FIRST)
    seconds, encoded = _preprocess(rt, proc, prompt)
    print(
        f"\na {len(prompt):,}-character prompt after {_VERIFY_FIRST} verified "
        f"ones: charged {seconds!r} s; the law over its {encoded} encoded ids "
        f"is {_law(encoded)!r} s"
    )
    assert encoded > 0
    assert seconds == pytest.approx(_law(encoded))


def test_a_verified_prompt_is_charged_its_segments_and_the_plain_check_once(rt, proc):
    prompt = _prompt(0)
    seconds, encoded = _preprocess(rt, proc, prompt)
    plain = len(proc.tokenizer.encode(prompt))
    assert seconds == pytest.approx(_law(encoded) + _law(plain))


def test_a_resent_prompt_is_charged_nothing_and_its_next_turn_its_new_text(rt, proc):
    for seed in range(_VERIFY_FIRST + 1):
        _preprocess(rt, proc, _prompt(seed))
    prompt = _prompt(_VERIFY_FIRST)
    assert _preprocess(rt, proc, prompt) == (0, 0)
    turn = prompt + "<|User|>" + _text(random.Random(-1), 30) + "<|Assistant|>"
    seconds, encoded = _preprocess(rt, proc, turn)
    assert 0 < encoded < 100
    assert seconds == pytest.approx(_law(encoded))


def test_on_a_real_run_the_backend_is_left_as_it_is(monkeypatch):
    monkeypatch.delenv(compass_run.ENV, raising=False)
    tok = _tokenizer()
    compass_run.tokenizer(tok, _config())
    assert vars(tok._tokenizer) == {}
