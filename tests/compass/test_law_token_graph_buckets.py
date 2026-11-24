# SPDX-License-Identifier: MIT
"""A runner that records prefill token-graph buckets holds the run's step linear
law to them as it states them at capture. A law's engine config states
``ATOM_TOKEN_GRAPH_SIZES`` only off ATOM's default, so a law fitted at the
former default and held in a store keys as a run at today's does; it states the
buckets in neither its engine config nor its prefill vector, installs, and is
refused at capture. A law whose prefill vector states the buckets this runner
records runs, one stating others is refused, those its records' runner
recorded at another batch budget too, which caps them and keys no law; an
engine config stating them is held to the run's at install, a prefill vector
beside it still to this runner's at capture, and a runner that records none
keeps every law.

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
#: Off ATOM's default, so an engine config states them.
STATED = "256,512,1024,2048,4096,8192"
REFUSED = "the step linear law states no ATOM_TOKEN_GRAPH_SIZES in its engine config"


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    """No performance variable and ATOM's default buckets; each test's token
    graphs go with it."""
    for name in (*PERFORMANCE_KEYS, BUCKETS):
        monkeypatch.delenv(name, raising=False)
    yield
    release_registered_graphs()


def recording(tmp_path, token_graphs=True, decode_only=False, **config):
    """The suite's deployment, with `config`'s fields, on an attention builder
    that records token graphs, or not, warming up at its batch budget, as
    ATOM's runner does at one DP rank without context parallelism."""
    r = deployment(tmp_path, disagg_is_decode=decode_only, **config)
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


def test_a_decode_only_runner_keeps_a_law_whose_prefill_vector_states_buckets(
    monkeypatch, tmp_path
):
    """A disaggregated decode engine records no prefill buckets, so a law whose
    records' runner did, as a prefill-and-decode deployment's, still runs."""
    r = recording(tmp_path, decode_only=True)
    run = build(tmp_path, "--step-linear", law_file(tmp_path, r, RECORDED))
    install(tmp_path, monkeypatch, run, r)

    assert r.capture_cudagraph() == (0.0, [], 0)


@pytest.mark.parametrize(
    "fitted_at, run_at",
    [(16384, 6000), (4000, 16384)],
    ids=["16384-6000", "4000-16384"],
)
def test_a_padded_law_recorded_at_another_batch_budget_is_refused(
    monkeypatch, tmp_path, fitted_at, run_at
):
    """`max_num_batched_tokens` caps the buckets ATOM records and keys no law,
    so a law recorded at another budget installs and is refused at capture."""
    r = recording(tmp_path, max_num_batched_tokens=run_at)
    theirs = [b for b in RECORDED if b <= fitted_at]
    run = build(tmp_path, "--step-linear", law_file(tmp_path, r, theirs))
    install(tmp_path, monkeypatch, run, r)

    with pytest.raises(ValueError, match=REFUSED) as e:
        r.capture_cudagraph()
    ours = [b for b in RECORDED if b <= run_at]
    assert str(e.value).endswith(
        f"and {theirs} as its prefill vector's buckets, and this run's runner "
        f"records the prefill token-graph buckets {ours}"
    )


def env_law(monkeypatch, tmp_path, fitted_at):
    """A runner at a batch budget of 6000 under `STATED`, and the run file of a
    law whose engine config states them and whose prefill vector is padded to
    those its records' runner recorded at `fitted_at`, installed on it; with
    `STATED` up to 6000 and up to `fitted_at`."""
    monkeypatch.setenv(BUCKETS, STATED)
    r = recording(tmp_path, max_num_batched_tokens=6000)
    stated = [int(b) for b in STATED.split(",")]
    theirs = [b for b in stated if b <= fitted_at]
    run = build(tmp_path, "--step-linear", law_file(tmp_path, r, theirs))
    install(tmp_path, monkeypatch, run, r)
    return r, [b for b in stated if b <= 6000], theirs


def test_an_engine_config_stating_the_buckets_refuses_a_law_recorded_at_another_budget(
    monkeypatch, tmp_path
):
    """The engine config, held to the run's at install, states ATOM's list; its
    records' runner recorded it up to a budget of 16384, this one to 6000."""
    r, ours, theirs = env_law(monkeypatch, tmp_path, 16384)

    with pytest.raises(ValueError) as e:
        r.capture_cudagraph()
    assert str(e.value) == (
        f"the step linear law states {BUCKETS}={STATED} in its engine config "
        f"and {theirs} as its prefill vector's buckets, and this run's runner "
        f"records the prefill token-graph buckets {ours}"
    )


def test_an_engine_config_stating_the_buckets_runs_a_law_recorded_at_the_runs_budget(
    monkeypatch, tmp_path
):
    r, ours, _ = env_law(monkeypatch, tmp_path, 6000)

    assert r.capture_cudagraph() == (0.0, [], 0)
    assert r.compass_backend.law.prefill.buckets == tuple(ours)
