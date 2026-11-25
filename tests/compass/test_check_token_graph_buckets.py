# SPDX-License-Identifier: MIT
"""The prefill token-graph buckets a run records, held where it starts and
before it does.

A host law's key states the graph sets its records' runner recorded, and a run
predicts its launch features under them, so a runner that records prefill
buckets refuses host laws whose key states others; a key stating none, or the
run's, keeps them, as do a runner recording none and an entry with no laws.

`compass run --check` works out the buckets the run's runner would record from
ATOM's arguments, its attention builder read where a device of the machine's
arch is declared, and lists a step linear law or host laws the run would refuse
at capture, with the runner's own refusal. A model whose builder records no
token graph, an eager run and a RapidServe one list nothing new.

No device: the check declares its machine's.
"""

from types import SimpleNamespace

import pytest
from test_drive_run import stub
from test_drive_runfile import LAWS, build, install, published, step_linear
from test_key_token_graph_sizes import BUCKETS, FORMER
from test_law_token_graph_buckets import (  # noqa: F401  fixture
    RECORDED,
    environment,
    law_file,
    recording,
)
from test_run_lacking import COVERED, checked, lacked, served
from test_run_step_grid import PLATFORM, deployment, host_laws_key

from atom.compass import run as compass_run
from atom.compass.artifacts import Kind, keyed_platform
from atom.compass.backends.launch_features import GRAPHS

V41 = "DeepseekV41ForCausalLM"
#: The former default's buckets, as a key's graph sets state them.
OTHERS = [int(b) for b in FORMER.split(",")]


def host_laws(tmp_path, r, buckets=None, laws=LAWS):
    """Host laws `laws` fitted under `r`'s host key, which states the graph sets
    of a V4.1 runner that recorded the prefill buckets `buckets`, or none."""
    key = {**compass_run.host_key(r.config), **keyed_platform(Kind.HOST_LAWS, PLATFORM)}
    if buckets is not None:
        token = {"prefill": buckets, "decode": []}
        graphs = {"architecture": V41, "token": token, "request": [], "decoder": None}
        key["engine"] = {**key["engine"], GRAPHS: graphs}
    params = {"key": key, "laws": laws}
    return published(tmp_path, r, Kind.HOST_LAWS, host_laws_key(r), params)


def v41(tmp_path, **config):
    """The suite's deployment with DeepSeek-V4.1's attention, whose builder
    records token graphs, on a runner stating that builder, as `recording`'s,
    and capture sizes whose rungs `step_linear`'s law covers."""
    r = deployment(tmp_path, capture_sizes=COVERED, disagg_is_decode=False, **config)
    r.config.hf_config.architectures = [V41]
    r.attn_metadata_builder = SimpleNamespace(
        step_needs_eager=lambda batch: False, supports_token_graph=True
    )
    r.warmup_max_tokens = lambda: r.config.max_num_batched_tokens
    return r


@pytest.mark.parametrize(
    "buckets, refused",
    [(OTHERS, f"buckets {OTHERS}, and this run's runner records {RECORDED}")]
    + [(RECORDED, None), (None, None)],
    ids=["other-buckets", "the-run-s-buckets", "no-graph-sets"],
)
def test_a_runner_recording_buckets_holds_host_laws_to_them(
    monkeypatch, tmp_path, buckets, refused
):
    r = recording(tmp_path)
    host = host_laws(tmp_path, r, buckets)
    run = build(
        tmp_path, "--step-stub", stub(tmp_path), "--host-laws", host.key.token()
    )
    install(tmp_path, monkeypatch, run, r)

    if refused is None:
        assert r.capture_cudagraph() == (0.0, [], 0)
    else:
        with pytest.raises(ValueError, match="the host laws' key states") as e:
            r.capture_cudagraph()
        assert str(e.value).endswith(refused)


@pytest.mark.parametrize(
    "decode_only, laws", [(True, LAWS), (False, {})], ids=["decode-only", "no-laws"]
)
def test_a_runner_recording_none_or_an_entry_with_no_laws_keeps_the_key(
    monkeypatch, tmp_path, decode_only, laws
):
    """A key stating other buckets is kept where nothing is predicted under it:
    on a runner that records none, and in an entry with no laws."""
    r = recording(tmp_path, decode_only=decode_only)
    host = host_laws(tmp_path, r, OTHERS, laws)
    run = build(
        tmp_path, "--step-stub", stub(tmp_path), "--host-laws", host.key.token()
    )
    install(tmp_path, monkeypatch, run, r)

    assert r.capture_cudagraph() == (0.0, [], 0)


@pytest.mark.parametrize("laws", ["step linear law", "host laws"])
def test_check_lists_what_a_v41_run_refuses_at_capture_with_its_refusal(
    monkeypatch, tmp_path, capsys, laws
):
    r = v41(tmp_path)
    if laws == "host laws":
        flags = ["--step-stub", stub(tmp_path)]
        flags += ["--host-laws", host_laws(tmp_path, r, OTHERS).key.token()]
    else:
        flags = ["--step-linear", step_linear(tmp_path, r).key.token()]
    served(monkeypatch, r.config)

    assert checked(tmp_path, *flags) == 1

    out = capsys.readouterr().out
    install(tmp_path, monkeypatch, build(tmp_path, *flags), r)
    with pytest.raises(ValueError) as refused:
        r.capture_cudagraph()
    assert lacked(out) == [f"{laws}: stale"]
    assert f"  refused: {refused.value}\n" in out
    assert str(RECORDED) in str(refused.value)


@pytest.mark.parametrize("states", ["prefill-vector", "engine-config"])
def test_check_keeps_a_law_stating_the_run_s_buckets(
    monkeypatch, tmp_path, capsys, states
):
    if states == "engine-config":
        monkeypatch.setenv(BUCKETS, FORMER)
    r = v41(tmp_path)
    law = law_file(tmp_path, r, RECORDED if states == "prefill-vector" else None)
    host = host_laws(tmp_path, r, RECORDED if states == "prefill-vector" else OTHERS)
    served(monkeypatch, r.config)

    assert checked(tmp_path, "--step-linear", law, "--host-laws", host.key.token()) == 0
    assert lacked(capsys.readouterr().out) == []


@pytest.mark.parametrize(
    "config",
    [None, {"enforce_eager": True}, {"enable_rapidserve": True}],
    ids=["no-token-graph-builder", "eager", "rapidserve"],
)
def test_a_run_recording_no_buckets_checks_as_before(
    monkeypatch, tmp_path, capsys, config
):
    """A V4.1 run records none eager, or under RapidServe, whose prefill engine
    runs eager and whose decode engine records no prefill."""
    if config is None:
        r = deployment(tmp_path, capture_sizes=COVERED)
    else:
        r = v41(tmp_path, **config)
    law = step_linear(tmp_path, r).key.token()
    host = host_laws(tmp_path, r, OTHERS).key.token()
    served(monkeypatch, r.config)

    assert checked(tmp_path, "--step-linear", law, "--host-laws", host) == 0
    assert lacked(capsys.readouterr().out) == []
