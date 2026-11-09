# SPDX-License-Identifier: MIT
"""The served tokenizer on a simulated run: ``encode`` and ``decode`` cost LP time.

`compass_run.tokenizer` is what the API server's ``main`` calls on the tokenizer
it serves. Here a run file's ``host`` entry states the `TERM` tokenizer term,
fitted on the fast backend, and the calls are made on the clock owner's thread,
as the event loop makes the final decode of a non-streaming completion.
"""

import ast
import hashlib
import json
import math
import pathlib
import time
import warnings
from types import SimpleNamespace

import pytest

from atom.compass import run as compass_run
from atom.compass.clock import LpId, single_engine_table
from atom.compass.run import step_grid
from atom.compass.spec.rules import FingerprintMismatch
from atom.utils import clock
from atom.utils.clock import LPRuntime

REPO = pathlib.Path(__file__).resolve().parents[2]
PROMPT_TOKENS, OUTPUT_TOKENS = 1000, 200
TERM = {
    "backend": "fast",
    "fingerprint": "sha256:" + "a" * 64,
    "encode": {"fixed_s": 3.0e-4, "tokens_per_s": 1.7e6},
    "encode_batch": {"fixed_s": 3.0e-4, "tokens_per_s": 1.7e6},
    "decode": {"fixed_s": 1.5e-4, "tokens_per_s": 2.55e6},
}
#: `TERM` with the segment tokenizer call's own work.
FULL = TERM | {"segment": {"fixed_s": 9.0e-4, "tokens_per_s": 4.0e6}}


class _Tokenizer:
    is_fast = True

    def encode(self, text):
        return list(range(len(text)))

    def decode(self, ids):
        return "x" * len(ids)


class _GrantAll:
    def send(self, msg):
        self.t = msg[1]

    def recv(self):
        return self.t, {}


def _run_file(monkeypatch, tmp_path, run: dict) -> dict:
    run = {"bound_s": 600.0} | run
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(path))
    return run


@pytest.fixture
def run_file(monkeypatch, tmp_path):
    _run_file(monkeypatch, tmp_path, {"host": {"tokenizer": TERM}})


@pytest.fixture
def rt():
    rt = LPRuntime(
        LpId("frontend"),
        single_engine_table(admission_s=0.54e-3, ipc_s=0.001, stream_s=0.002),
        _GrantAll(),
    )
    clock.install(rt)
    rt.start_run()
    yield rt
    clock.install(None)


def _charges(tok, rt) -> tuple[float, float]:
    """The LP seconds one request's encode and its output's decode cost."""
    ids = tok.encode("p" * PROMPT_TOKENS)
    encoded = rt.now
    assert tok.decode(ids[:OUTPUT_TOKENS]) == "x" * OUTPUT_TOKENS
    return encoded, rt.now - encoded


def test_on_a_run_encode_and_decode_advance_the_lp_clock(run_file, rt):
    tok = _Tokenizer()
    compass_run.tokenizer(tok, None)
    encoded, decoded = _charges(tok, rt)
    print(f"\none request: encode {encoded!r} s, decode {decoded!r} s of LP time")
    encode, decode = TERM["encode"], TERM["decode"]
    assert encoded == pytest.approx(
        encode["fixed_s"] + PROMPT_TOKENS / encode["tokens_per_s"]
    )
    assert decoded == pytest.approx(
        decode["fixed_s"] + OUTPUT_TOKENS / decode["tokens_per_s"]
    )


def test_with_no_run_file_the_tokenizer_is_left_as_it_is(monkeypatch):
    monkeypatch.delenv(compass_run.ENV, raising=False)
    tok = _Tokenizer()
    compass_run.tokenizer(tok, None)
    assert vars(tok) == {}


@pytest.mark.parametrize("run", [{"host": {"laws": {}}}, {}], ids=["entry", "none"])
def test_with_no_tokenizer_term_the_tokenizer_is_left_as_it_is(
    monkeypatch, tmp_path, rt, run
):
    _run_file(monkeypatch, tmp_path, run)
    tok = _Tokenizer()
    compass_run.tokenizer(tok, None)
    assert vars(tok) == {} and _charges(tok, rt) == (0.0, 0.0)


def test_a_term_fitted_on_another_backend_is_refused(run_file):
    tok = _Tokenizer()
    tok.is_fast = False
    with pytest.raises(
        ValueError,
        match="fitted on the fast backend and the server loaded the slow one",
    ):
        compass_run.tokenizer(tok, None)


@pytest.mark.parametrize(
    "term",
    [
        {k: v for k, v in TERM.items() if k != "fingerprint"},
        TERM | {"backend": "rust"},
        TERM | {"encode": {"fixed_s": 3.0e-4}},
        TERM | {"decode": {"fixed_s": 1.5e-4, "tokens_per_s": 0.0}},
        TERM | {"derate": 1.0},
    ],
    ids=["no-fingerprint", "other-backend-name", "no-rate", "zero-rate", "derate"],
)
def test_a_tokenizer_term_a_charge_would_misread_is_refused(
    monkeypatch, tmp_path, term
):
    _run_file(monkeypatch, tmp_path, {"host": {"tokenizer": term}})
    with pytest.raises(ValueError, match="host's tokenizer is"):
        compass_run.spec()


def test_each_kind_of_call_is_charged_its_own_term_a_flat_decode_inside_its_tokens(
    monkeypatch, tmp_path, rt
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    several = {"fixed_s": 1.0e-3, "tokens_per_s": 1.0e4}
    flat = {"fixed_s": 2.0e-5, "tokens": [1, 9]}
    term = TERM | {"encode_batch": several, "decode": flat}
    _run_file(monkeypatch, tmp_path, {"host": {"tokenizer": term}})
    tok = _Tokenizer()
    tok._tokenizer = Tokenizer(WordLevel({"a": 0}, unk_token="a"))
    tok._tokenizer.pre_tokenizer = Whitespace()
    compass_run.tokenizer(tok, None)

    def cost(call, *args):
        start = rt.now
        call(*args)
        return rt.now - start

    one = cost(tok._tokenizer.encode_batch, ["a a"])
    assert one == pytest.approx(TERM["encode"]["fixed_s"] + 2 / 1.7e6)
    batch = cost(tok._tokenizer.encode_batch, ["a a", "a a a"])
    assert batch == pytest.approx(1.0e-3 + 5 / 1.0e4)
    assert cost(tok.decode, [0] * 9) == pytest.approx(2.0e-5)
    assert cost(tok.decode, [0] * 10) == 0.0
    assert rt.refusals == [
        "tokenizer:decode of 10 tokens, outside the 1 to 9 tokens its decode was "
        + "fitted on"
    ]


def test_a_segment_tokenizer_call_is_charged_its_own_work_over_the_ids_it_encoded(
    monkeypatch, tmp_path, rt
):
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from atom.model_engine import segment_tokenizer

    cls = segment_tokenizer.SegmentTokenizer
    monkeypatch.setattr(cls, "_encode_cached", vars(cls)["_encode_cached"])
    # Every prompt takes the cache, and none is checked with a plain encode.
    monkeypatch.setattr(segment_tokenizer, "_MIN_CACHED_CHARS", 0)
    monkeypatch.setattr(segment_tokenizer, "_VERIFY_FIRST", 0)
    several = {"fixed_s": 1.0e-3, "tokens_per_s": 1.0e4}
    term = FULL | {"encode_batch": several}
    _run_file(monkeypatch, tmp_path, {"host": {"tokenizer": term}})
    backend = Tokenizer(WordLevel({"a": 0, "b": 1, "c": 2}, unk_token="a"))
    backend.pre_tokenizer = Whitespace()
    backend.add_special_tokens([AddedToken("<sep>", special=True)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend)
    seg = cls.maybe_create(tok, 1 << 20)  # the engine is built before the wrap
    compass_run.tokenizer(tok, None)

    def cost(prompt):
        start = rt.now
        assert seg.encode(prompt).tolist() == backend.encode(prompt).ids
        return rt.now - start

    own, one = term["segment"], term["encode"]
    # Two segments of five ids in one encode_batch, then none, then one of one id.
    assert cost("a a a<sep>b b") == pytest.approx(
        several["fixed_s"] + 5 / 1.0e4 + own["fixed_s"] + 5 / own["tokens_per_s"]
    )
    assert cost("a a a<sep>b b") == pytest.approx(own["fixed_s"])
    assert cost("a a a<sep>b b<sep>c") == pytest.approx(
        one["fixed_s"] + 1 / one["tokens_per_s"] + own["fixed_s"] + 1 / 4.0e6
    )


class _LoadedFrom(_Tokenizer):
    """A fast tokenizer loaded from a model directory, as ``from_pretrained``
    records it."""

    def __init__(self, directory):
        self.name_or_path = str(directory)


def _model_dir(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text('{"version": "1.0"}')
    return model, "sha256:" + hashlib.sha256(b'{"version": "1.0"}').hexdigest()


def test_a_tokenizer_json_other_than_the_fitted_one_warns(run_file, tmp_path):
    model, loaded = _model_dir(tmp_path)
    with pytest.warns(FingerprintMismatch) as caught:
        compass_run.tokenizer(_LoadedFrom(model), None)
    message = str(caught[0].message)
    print(f"\n{message}")
    assert TERM["fingerprint"] in message and loaded in message


def test_the_fitted_tokenizer_json_is_silent(monkeypatch, tmp_path):
    model, loaded = _model_dir(tmp_path)
    term = TERM | {"fingerprint": loaded}
    _run_file(monkeypatch, tmp_path, {"host": {"tokenizer": term}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compass_run.tokenizer(_LoadedFrom(model), None)


def test_a_server_charges_the_entry_fitted_under_its_engine_config(
    monkeypatch, tmp_path, rt
):
    # Two entries, one per engine config; this server's config keys the second.
    slow = {"fixed_s": 1.0, "tokens_per_s": 1.0}
    intake = {"fixed_s": 2.0e-3, "per_byte_s": 4.0e-9}
    hosts = [
        {"key": {"engine": {"side": "P"}}, "tokenizer": TERM | {"encode": slow}},
        {"key": {"engine": {"side": "D"}}, "tokenizer": TERM, "http_intake": intake},
    ]
    _run_file(monkeypatch, tmp_path, {"host": hosts})
    monkeypatch.setattr(compass_run, "_frontend_host", None)
    with pytest.raises(ValueError, match="none was resolved for this server's"):
        compass_run.http_intake()
    monkeypatch.setattr(step_grid, "keyed", lambda config: {"side": "D"})
    tok = _Tokenizer()
    compass_run.tokenizer(tok, None)
    encoded, _ = _charges(tok, rt)
    assert encoded == pytest.approx(
        TERM["encode"]["fixed_s"] + PROMPT_TOKENS / TERM["encode"]["tokens_per_s"]
    )
    assert compass_run.http_intake() == (2.0e-3, 4.0e-9, 2.0e-3)


@pytest.mark.parametrize(
    "host, named",
    [
        (None, ["http_intake", "tokenizer"]),
        ({"tokenizer": FULL}, ["http_intake"]),
        (
            {"tokenizer": {k: v for k, v in FULL.items() if k != "encode_batch"}},
            ["http_intake", "tokenizer.encode_batch"],
        ),
        ({"tokenizer": TERM}, ["http_intake", "tokenizer.segment"]),
        ({"tokenizer": FULL, "http_intake": {"fixed_s": 0.0, "per_byte_s": 0.0}}, []),
    ],
    ids=["no-entry", "no-intake", "no-batch", "no-segment", "both"],
)
def test_the_summary_names_each_term_the_run_charged_nothing_for(
    monkeypatch, tmp_path, host, named
):
    run = {
        "admission_s": 0.54e-3,
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "clock_endpoint": "inproc:unused",
        "out_dir": str(tmp_path),
    }
    run = _run_file(monkeypatch, tmp_path, run | ({"host": host} if host else {}))
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()
    monkeypatch.setattr(compass_run, "_authority", authority)
    compass_run._hand_in(run, compass_run.ENGINE.name, [])
    frontend = LPRuntime(compass_run.FRONTEND, compass_run.channel_table(run), None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    print(f"\n  uncharged: {summary['uncharged_host_terms']}")
    assert summary["uncharged_host_terms"] == named


@pytest.mark.parametrize(
    "law", sorted(p.name for p in (REPO / "scripts/compass/laws").glob("*.json"))
)
def test_every_committed_law_file_charges_both_terms(monkeypatch, tmp_path, law):
    host = json.loads((REPO / "scripts/compass/laws" / law).read_text())["host"]
    _run_file(monkeypatch, tmp_path, {"host": host})
    compass_run.spec()
    assert compass_run.http_intake() is not None
    assert host["tokenizer"]["backend"] == "fast"


def test_the_server_wraps_its_tokenizer_after_the_delta_reuse_probe():
    """The probe decodes at startup, before the run has traffic; a charged
    decode there waits on a clock that no request will advance."""
    tree = ast.parse((REPO / "atom/entrypoints/openai/api_server.py").read_text())
    main = next(
        f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == "main"
    )
    body = [ast.unparse(s) for s in main.body]

    def first(text):
        return next(i for i, s in enumerate(body) if text in s)

    assert first("enable_delta_reuse(") < first("compass_run.tokenizer(")
