# SPDX-License-Identifier: MIT
"""A simulated deployment's frontends share their segment token ids.

Two API servers of one deployment are built on the same small trained
tokenizer, each with its own `InputOutputProcessor` and segment tokenizer, as
two prefill instances are; the run file names an ``out_dir``. What the first
encodes, the second reads back: same ids, same charges, and its backend sees
no segment text again.
"""

import json
from pathlib import Path

import pytest
from test_segment_tokenizer_charge import _config, _preprocess, _prompt, _tokenizer
from test_served_tokenizer import TERM
from tokenizers import AddedToken, Encoding

from atom.compass import run as compass_run
from atom.compass.clock import LpId, single_engine_table
from atom.compass.run import token_ids
from atom.model_engine.llm_engine import InputOutputProcessor
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.clock import LPRuntime


class _GrantAll:
    def send(self, msg):
        self.t = msg[1]

    def recv(self):
        return self.t, {}


@pytest.fixture
def out(tmp_path) -> Path:
    return tmp_path / "out"


@pytest.fixture
def rt(monkeypatch, tmp_path, out):
    path = tmp_path / "run.json"
    run = {"bound_s": 600.0, "host": {"tokenizer": TERM}, "out_dir": str(out)}
    path.write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(path))
    monkeypatch.delenv(token_ids.ENV, raising=False)
    rt = LPRuntime(
        LpId("frontend"),
        single_engine_table(admission_s=0.54e-3, ipc_s=0.001, stream_s=0.002),
        _GrantAll(),
    )
    clock.install(rt)
    rt.start_run()
    yield rt
    clock.install(None)


def _server(seen=None):
    """One API server's tokenizer and processor, wrapped as a simulated
    server's; `seen` collects each segment text its backend is asked for."""
    tok, config = _tokenizer(), _config()
    proc = InputOutputProcessor(config, tok, 16)
    if seen is not None:
        backend = tok._tokenizer
        raw = backend.encode_batch

        def counted(*args, **kwargs):
            if kwargs == {"add_special_tokens": False}:
                seen.extend(args[0])
            return raw(*args, **kwargs)

        backend.encode_batch = counted
    compass_run.tokenizer(tok, config)
    return proc


def test_a_second_server_reads_back_what_the_first_encoded_at_the_same_charge(
    rt, out, monkeypatch
):
    monkeypatch.setattr(token_ids, "MIN_CHARS", 0)
    prompt = _prompt(0)
    first = _server()
    charged = _preprocess(rt, first, prompt)
    ids = first.tokenizer.encode(prompt)
    assert any(out.glob("token_ids/*/*/*"))
    seen = []
    second = _server(seen)
    assert _preprocess(rt, second, prompt) == charged
    assert seen == []
    assert list(second.preprocess(prompt, SamplingParams()).token_ids) == ids


def test_a_short_text_is_encoded_and_not_kept(tmp_path):
    backend = _tokenizer()._tokenizer
    token_ids.install(backend, tmp_path)
    short = "alpha beta gamma"
    (got,) = backend.encode_batch([short], add_special_tokens=False)
    assert isinstance(got, Encoding) and not any(tmp_path.rglob("*"))


def test_a_long_text_is_kept_once_and_read_back_with_its_own_ids(tmp_path):
    backend = _tokenizer()._tokenizer
    plain = backend.encode_batch
    token_ids.install(backend, tmp_path)
    text = _prompt(1)[: token_ids.MIN_CHARS * 2]
    (cold,) = backend.encode_batch([text], add_special_tokens=False)
    (warm,) = backend.encode_batch([text], add_special_tokens=False)
    assert len(list(tmp_path.rglob("*/*/*"))) == 1
    assert not isinstance(warm, Encoding)
    assert list(warm.ids) == cold.ids == plain([text], add_special_tokens=False)[0].ids


def test_any_other_call_goes_to_the_backend_untouched(tmp_path):
    backend = _tokenizer()._tokenizer
    token_ids.install(backend, tmp_path)
    text = _prompt(2)[: token_ids.MIN_CHARS * 2]
    (pretokenized,) = backend.encode_batch(
        [text], add_special_tokens=False, is_pretokenized=False
    )
    (special,) = backend.encode_batch([text])
    backend.enable_truncation(64)
    (truncated,) = backend.encode_batch([text], add_special_tokens=False)
    assert all(isinstance(e, Encoding) for e in (pretokenized, special, truncated))
    assert len(truncated.ids) == 64 and not any(tmp_path.rglob("*"))


def test_another_tokenizer_keeps_its_ids_apart(tmp_path):
    one, other = _tokenizer()._tokenizer, _tokenizer()._tokenizer
    other.add_special_tokens([AddedToken("<|extra|>", special=True)])
    token_ids.install(one, tmp_path)
    token_ids.install(other, tmp_path)
    text = _prompt(3)[: token_ids.MIN_CHARS * 2]
    one.encode_batch([text], add_special_tokens=False)
    other.encode_batch([text], add_special_tokens=False)
    assert len({p.parent.parent for p in tmp_path.rglob("*/*/*")}) == 2


def test_another_tokenizers_build_keeps_its_ids_apart(tmp_path, monkeypatch):
    import tokenizers

    text = _prompt(4)[: token_ids.MIN_CHARS * 2]
    for version in ("0.0.1", "0.0.2"):
        monkeypatch.setattr(tokenizers, "__version__", version)
        backend = _tokenizer()._tokenizer
        token_ids.install(backend, tmp_path)
        backend.encode_batch([text], add_special_tokens=False)
    assert len({p.parent.parent for p in tmp_path.rglob("*/*/*")}) == 2


def test_the_ids_go_under_the_run_s_out_dir_or_where_the_variable_says(
    monkeypatch, tmp_path
):
    run = {"out_dir": str(tmp_path / "out")}
    monkeypatch.delenv(token_ids.ENV, raising=False)
    assert token_ids.directory(run) == tmp_path / "out" / token_ids.SUBDIR
    assert token_ids.directory({}) is None
    monkeypatch.setenv(token_ids.ENV, str(tmp_path / "shared"))
    assert token_ids.directory(run) == tmp_path / "shared"
    monkeypatch.setenv(token_ids.ENV, token_ids.OFF)
    assert token_ids.directory(run) is None
