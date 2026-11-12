# SPDX-License-Identifier: MIT
"""A runner that records prefill token-graph buckets holds the run's step linear
law to them as it states them at capture. A law's engine config states
``ATOM_TOKEN_GRAPH_SIZES`` only off ATOM's default, so a law fitted at the
former default and held in a store keys as a run at today's does; it states the
buckets in neither its engine config nor its prefill vector, installs, and is
refused at capture. A law whose prefill vector states the buckets this runner
records runs, one stating others is refused, an engine config stating them is
held to the run's at install, and a runner that records none keeps every law.

No device.
"""

import json
from types import SimpleNamespace

import pytest
from test_backend_step_linear import law
from test_drive_runfile import build, install, step_linear
from test_key_token_graph_sizes import BUCKETS, FORMER
from test_run_step_grid import deployment
from test_step_linear_token_padding import PADDED

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.backends import StepLinearBackend
from atom.compass.run.step_grid import keyed
from atom.utils.graph_holders import release_registered_graphs

#: The buckets ATOM's default records up to the deployment's batch budget, 8192.
RECORDED = [256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168]
REFUSED = "the step linear law states no ATOM_TOKEN_GRAPH_SIZES in its engine config"


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    """No performance variable and ATOM's default buckets; each test's token
    graphs go with it."""
    for name in (*PERFORMANCE_KEYS, BUCKETS):
        monkeypatch.delenv(name, raising=False)
    yield
    release_registered_graphs()


def recording(tmp_path, token_graphs=True, decode_only=False):
    """The suite's deployment on an attention builder that records token
    graphs, or not, warming up at its batch budget, as ATOM's runner does at
    one DP rank without context parallelism."""
    r = deployment(tmp_path, disagg_is_decode=decode_only)
    r.attn_metadata_builder = SimpleNamespace(
        step_needs_eager=lambda batch: False, supports_token_graph=token_graphs
    )
    r.warmup_max_tokens = lambda: r.config.max_num_batched_tokens
    return r


def law_file(tmp_path, r, buckets=None):
    """A law file fitted under `r`'s engine config, its prefill vector padded
    to `buckets` when given, else with no padding and no buckets."""
    stated = law()
    if buckets is not None:
        hull = {"equal": dict.fromkeys(list(PADDED)[1:], 0), "facets": []}
        stated["prefill"] = {"coefficients": PADDED, "hull": hull, "buckets": buckets}
    path = tmp_path / "law.json"
    path.write_text(json.dumps({**stated, "engine_config": keyed(r.config)}))
    return str(path)


def test_a_store_law_stating_no_buckets_installs_and_is_refused_at_capture(
    monkeypatch, tmp_path
):
    r = recording(tmp_path)
    run = build(tmp_path, "--step-linear", step_linear(tmp_path, r).key.token())
    install(tmp_path, monkeypatch, run, r)
    assert isinstance(r.compass_backend, StepLinearBackend)

    with pytest.raises(ValueError, match=REFUSED) as e:
        r.capture_cudagraph()
    assert str(e.value).endswith(
        "and none as its prefill vector's buckets, and this run's runner records "
        f"the prefill token-graph buckets {RECORDED}"
    )


def test_a_law_whose_prefill_vector_states_the_former_buckets_is_refused(
    monkeypatch, tmp_path
):
    r = recording(tmp_path)
    former = [int(b) for b in FORMER.split(",")]
    run = build(tmp_path, "--step-linear", law_file(tmp_path, r, former))
    install(tmp_path, monkeypatch, run, r)

    with pytest.raises(ValueError, match=REFUSED) as e:
        r.capture_cudagraph()
    assert f"and {former} as its prefill vector's buckets" in str(e.value)


def test_a_law_whose_prefill_vector_states_the_buckets_recorded_runs(
    monkeypatch, tmp_path
):
    r = recording(tmp_path)
    run = build(tmp_path, "--step-linear", law_file(tmp_path, r, RECORDED))
    install(tmp_path, monkeypatch, run, r)

    assert r.capture_cudagraph() == (0.0, [], 0)
    assert r.compass_backend.law.prefill.buckets == tuple(RECORDED)


def test_an_engine_config_stating_the_buckets_holds_the_law_to_them(
    monkeypatch, tmp_path
):
    """A law whose records ran the former buckets and whose engine config says
    so runs on a runner that sets them, its prefill vector stating none."""
    monkeypatch.setenv(BUCKETS, FORMER)
    r = recording(tmp_path)
    run = build(tmp_path, "--step-linear", law_file(tmp_path, r))
    install(tmp_path, monkeypatch, run, r)

    assert r.capture_cudagraph() == (0.0, [], 0)
    assert run["step_linear"]["engine_config"]["env"] == {BUCKETS: FORMER}


@pytest.mark.parametrize(
    "token_graphs, decode_only",
    [(False, False), (True, True)],
    ids=["no-token-graph-builder", "decode-only"],
)
def test_a_runner_that_records_no_token_graph_keeps_a_law_stating_none(
    monkeypatch, tmp_path, token_graphs, decode_only
):
    r = recording(tmp_path, token_graphs, decode_only)
    run = build(tmp_path, "--step-linear", step_linear(tmp_path, r).key.token())
    install(tmp_path, monkeypatch, run, r)

    assert r.capture_cudagraph() == (0.0, [], 0)
