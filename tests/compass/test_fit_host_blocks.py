# SPDX-License-Identifier: MIT
"""`compass fit host`'s opt-in host fit of a theory-designed calibration
(`fit.host_blocks`): one constant per launch path held out in blocks per path
kind, decode client runs fitted on their drains alone, the tokenizer and intake
terms on the steady windows, and a run that prices the path constants.

Decode forwards replay a graph per rung, each rung's launch its own constant
(`RUNG_S`): rung 1 the commonest, rung 2 too rare for a constant of its own,
rung 8 unseen. A planted provider's prefill past `BUCKET` tokens runs its body
eagerly, `EAGER_S` dearer.
"""

import dataclasses
import json

import numpy as np
import pytest
from test_backend_step_grid import scheduler_of
from test_fit_host import host_laws_key, stamped
from test_fit_steps import (  # noqa: F401
    HOST,
    PLATFORM,
    clients,
    fit,
    member,
    planted_roots,
    record,
)
from test_memory_readings import META_8B
from test_parity_recurrence import _labelled, _law
from test_parity_recurrence import record as recurrence_record
from test_run_step_grid import deployment, publish, run_file

from atom.compass import run as compass_run
from atom.compass.artifacts import Kind, keyed_platform
from atom.compass.backends.launch_features import (
    GRAPHS,
    PATHS,
    REGISTRY,
    LaunchFeatures,
    path,
)
from atom.compass.backends.shape import BatchView, RequestShape, sum_tokens
from atom.compass.backends.step_labels import Sample
from atom.compass.backends.step_linear_fit import (
    fit_host,
    label_records,
)
from atom.compass.fit import host_blocks
from atom.compass.fit.host_blocks import (
    BLOCKS,
    KIND_BLOCKS_ENV,
    PATH_MIN,
    drain_cut,
    fill_flags,
    fill_rungs,
    price,
    refit,
)
from atom.compass.parity import recurrence
from atom.compass.runner.overrides import (
    FORWARD_KINDS,
    HOST_SEGMENTS,
    HostDeviceRecurrence,
)

ARCH = "PathsForCausalLM"
BUCKET = 4000
SETS = {"architecture": ARCH, "token": {"prefill": [BUCKET], "decode": []}}
EAGER_S = 0.04
#: Each decode rung's launch constant; rung 8 runs no forward.
RUNG_S = {1: 2.0e-3, 2: 2.2e-3, 4: 2.4e-3, 16: 2.9e-3}
LADDER = [1, 2, 4, 8, 16]
#: The forwards each rung runs: rung 2 fewer than `PATH_MIN`, and fewer than
#: `BLOCKS`, so some held-out block lacks it.
RUNS = {1: 160, 2: 3, 4: 80, 16: 60}


class Paths(LaunchFeatures):
    features = ("eager",)

    def predict(self, view, graphs):
        bucket = max(graphs["token"]["prefill"])
        return {
            "eager": float(bool(view.prefill) and sum_tokens(view.requests) > bucket)
        }


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    monkeypatch.setitem(REGISTRY, ARCH, Paths())


def _key(ladder=LADDER):
    return {"engine": {"ladder": ladder, GRAPHS: SETS}}


def _decodes(seed=3720, key=None):
    """Decode forwards in rung order cycling through `RUNS`, each launch its
    rung's constant plus 1 us a token, within 1%."""
    rng, out = np.random.default_rng(seed), []
    key = _key() if key is None else key
    order = [r for r, k in RUNS.items() for _ in range(k)]
    rng.shuffle(order)
    for n, rung in enumerate(order):
        seqs = int(rng.integers(max(rung // 2, 1), rung + 1))
        view = BatchView(
            (RequestShape(6, 1000, True),) * seqs,
            capture_rung=rung,
            produces_output=True,
        )
        host = {g: 1e-4 * rng.uniform(0.99, 1.01) for g in HOST_SEGMENTS}
        w = RUNG_S[rung] + 1e-6 * sum_tokens(view.requests)
        host["W_launch"] = w * rng.uniform(0.99, 1.01)
        provenance = {"source": "record", "run": "a", "requests": [n], "config": key}
        launch = Paths().predict(view, SETS)
        out.append(
            Sample(n, 0, view, 0.01, None, host, False, provenance, launch=launch)
        )
    return out


def _prefills(seed=3721, start=0):
    """Output prefills, a third past `BUCKET` tokens, `EAGER_S` dearer to launch."""
    rng, out = np.random.default_rng(seed), []
    for i in range(90):
        tokens = int(
            rng.integers(BUCKET + 1, 6000) if i % 3 == 0 else rng.integers(256, BUCKET)
        )
        view = BatchView((RequestShape(tokens, tokens, False),), produces_output=True)
        launch = Paths().predict(view, SETS)
        host = {g: 1e-4 * rng.uniform(0.99, 1.01) for g in HOST_SEGMENTS}
        w = 2e-3 + 1e-6 * tokens + EAGER_S * launch["eager"]
        host["W_launch"] = w * rng.uniform(0.99, 1.01)
        provenance = {"source": "record", "run": "a", "requests": [i], "config": _key()}
        out.append(
            Sample(
                start + i, 0, view, 0.01, None, host, False, provenance, launch=launch
            )
        )
    return out


def test_a_forward_s_launch_path_is_its_rung_or_its_eager_features():
    decode = BatchView(
        (RequestShape(6, 100, True),), capture_rung=4, produces_output=True
    )
    assert path(decode, {}) == "rung4"
    eager = BatchView((RequestShape(6, 100, True),), produces_output=True)
    assert path(eager, {"split": 0.0}) == "eager"
    prefill = BatchView((RequestShape(900, 900, False),), produces_output=True)
    assert path(prefill, {"eager_body": 0.0, "eager_tail": 0.0}) == "replay"
    assert (
        path(prefill, {"eager_tail": 1.0, "eager_body": 1.0}) == "eager_body+eager_tail"
    )
    middle = BatchView((RequestShape(900, 900, False),), produces_output=False)
    assert path(middle, {"eager_body": 1.0}) == "eager_body"


def test_each_rung_takes_its_own_launch_constant_and_a_rare_or_unseen_one_is_filled():
    samples = _decodes()
    host, report = fit_host(samples)
    loop = {g: host["laws"]["decode"][g] for g in HOST_SEGMENTS if g != "W_launch"}
    refit(samples, host, report)

    law, own = host["laws"]["decode"]["W_launch"], host[PATHS]["decode"]["W_launch"]
    fitted = report["decode.W_launch"]
    assert fitted["base"] == "rung1" and fitted["short"] == ["rung2"]
    assert law[0] == pytest.approx(RUNG_S[1], rel=0.02)
    for rung in (4, 16):
        assert own[f"rung{rung}"] == pytest.approx(RUNG_S[rung], rel=0.02)
    # Rung 2 lies a third of the way from rung 1 to rung 4; rung 8 between 4 and 16.
    assert own["rung2"] == pytest.approx(law[0] + (own["rung4"] - law[0]) / 3)
    assert own["rung8"] == pytest.approx(
        own["rung4"] + (own["rung16"] - own["rung4"]) / 3
    )
    assert fitted["filled"] == {"rung2": own["rung2"], "rung8": own["rung8"]}
    # Every path is held out in every block; rung 2's three forwards are left
    # out of the fit and its blocks alike.
    held = fitted["held_out"]
    assert len(held["blocks"]) == BLOCKS
    assert {p: v["blocks"] for p, v in held["paths"].items()} == {
        "rung1": BLOCKS,
        "rung4": BLOCKS,
        "rung16": BLOCKS,
    }
    assert held["error"] < 0.01 < held["error_pooled"]
    assert fitted["choice"] == {
        "paths": held["error"],
        "fit_host": pytest.approx(held["error_pooled"]),
        "kept": False,
    }
    checks = report["kind_blocks"]
    assert checks["H1"]["decode.W_launch"] == {"short": ["rung2"], "pass": False}
    assert checks["H2"]["decode.W_launch"] == {"short": ["rung2"], "pass": False}
    assert not checks["H1"]["pass"] and PATH_MIN > RUNS[2]
    # The engine loop's segments keep their law, validated in the same blocks.
    assert {g: host["laws"]["decode"][g] for g in loop} == loop
    assert report["decode.E"]["held_out"]["paths"]["rung16"]["blocks"] == BLOCKS
    assert set(host[PATHS]["decode"]) == {"W_launch"}
    # A run prices a forward at its rung's constant.
    x = (1, 6 * 8, 8)
    assert price(host, "decode", "W_launch", x, "rung16") == pytest.approx(
        own["rung16"] + law[1] * 48
    )
    assert price(host, "decode", "W_launch", x, "eager") == pytest.approx(
        law[0] + law[1] * 48
    )


def test_a_prefill_s_eager_path_is_priced_its_own_constant_or_its_column():
    """One feature: a constant per path spans what an intercept and the
    column do, so both laws price each path alike and hold out alike; the tie
    keeps the law `fit_host` gave, its column."""
    samples = _prefills()
    host, report = fit_host(samples)
    assert host["columns"]["prefill"]["W_launch"] == {
        "eager": pytest.approx(EAGER_S, rel=0.05)
    }
    refit(samples, host, report)

    x = (1, 5000, 1)
    eager = price(host, "prefill", "W_launch", x, "eager", {"eager": 1.0})
    replay = price(host, "prefill", "W_launch", x, "replay", {"eager": 0.0})
    assert eager == pytest.approx(2e-3 + 5e-3 + EAGER_S, rel=0.02)
    assert replay == pytest.approx(2e-3 + 5e-3, rel=0.02)
    choice = report["prefill.W_launch"]["choice"]
    assert choice["kept"] and choice["paths"] == pytest.approx(choice["fit_host"])
    assert PATHS not in host and "eager" in host["columns"]["prefill"]["W_launch"]
    assert report["kind_blocks"]["H1"]["prefill.W_launch"]["pass"]


def test_a_launch_no_rung_moves_keeps_the_law_fit_host_gave():
    samples = [
        dataclasses.replace(s, host=s.host | {"W_launch": s.host["W_pre"] * 20})
        for s in _decodes()
    ]
    host, report = fit_host(samples)
    law = host["laws"]["decode"]["W_launch"]
    refit(samples, host, report)

    assert host["laws"]["decode"]["W_launch"] == law and PATHS not in host
    entry = report["decode.W_launch"]
    assert entry["choice"]["kept"] and "held_out" in entry["paths_tried"]
    assert entry["held_out"]["paths"]["rung4"]["blocks"] == BLOCKS


def test_a_path_too_rare_for_its_own_constant_keeps_the_additive_price():
    own = {"eager_tail": 0.05}
    filled = fill_flags(
        0.01,
        {"eager_body": 0.03},
        ("eager_body", "eager_tail"),
        "eager_body+eager_tail",
        own,
    )
    assert filled == {"replay": 0.01, "eager_body": 0.04}
    assert own == {"eager_tail": 0.05, "replay": 0.01, "eager_body": 0.04}
    rungs = {"rung4": 4.0}
    assert fill_rungs(1.0, "rung1", rungs, [1, 2, 4, 8]) == {"rung2": 2.0, "rung8": 4.0}


def test_a_decode_client_run_gives_only_its_drain_s_samples(monkeypatch):
    samples = _decodes()[:10]
    monkeypatch.setattr(host_blocks, "record_name", lambda record: "a")
    runs = {"rec": [("export", {"span": [2, 8], "window": [0, 1], "prefills": [2, 3]})]}
    kept, report = drain_cut(samples, runs, {("rec", "export")})
    assert [s.step for s in kept] == [0, 1, 4, 5, 6, 7, 8, 9]
    assert report == [
        {
            "record": "rec",
            "export": "export",
            "span": [2, 8],
            "drains": 1,
            "drain_steps": 5,
            "dropped": 2,
            "kept": 5,
        }
    ]


def test_the_command_fits_a_decode_client_run_on_its_drains_and_states_the_checks(
    tmp_path, monkeypatch
):
    a = record(tmp_path / "a", 19)
    stamped(a)
    monkeypatch.setenv(KIND_BLOCKS_ENV, "1")
    fit(tmp_path, "fit host", a, "--decode-client-run", a, a / "client.json")

    host = member(tmp_path / "store", host_laws_key(HOST), "params")
    report = member(tmp_path / "store", host_laws_key(HOST), "report")
    # The record alternates prefill and decode steps: the drains hold its decodes.
    samples = label_records([a / "steps/dp0.jsonl"])[0]
    decodes = [s for s in samples if not s.view.prefill]
    (run,) = report["decode_client_runs"]
    assert run["dropped"] == len(samples) - len(decodes) and run["kept"] == len(decodes)
    assert set(host["laws"]) == {"decode"}
    alone, _ = fit_host(decodes)
    refit(decodes, alone, {"unfitted": {}})
    assert host["laws"]["decode"] == alone["laws"]["decode"]
    checks = report["kind_blocks"]
    assert {"H1", "H2", "H3", "H4", "H5", "H3_pass", "H5_pass"} <= set(checks)
    # Not graded-only, so not held to steady windows alone; the decode run is no
    # steady window, so its stamps give no tokenizer or intake term.
    assert checks["H4"] == {"graded_only": False, "decode_drains": 1, "pass": False}
    assert "http_intake" not in host and "http_intake" in report["unfitted"]


def test_a_recurrence_prices_a_path_constant_in_place_of_its_law_s():
    laws = {k: {g: (1e-4, 0.0, 0.0) for g in HOST_SEGMENTS} for k in FORWARD_KINDS}
    paths = {"decode": {"W_launch": {"rung16": 3e-3}}}
    rec = HostDeviceRecurrence(laws, (), lambda: 0.0, paths=paths)
    x = (1, 96, 16)
    assert rec.prepare_end(0.0, "decode", x, path="rung16")[1][
        "W_launch"
    ] == pytest.approx(3e-3)
    assert rec.prepare_end(0.0, "decode", x, path="rung8")[1]["W_launch"] == 1e-4
    assert rec.prepare_end(0.0, "decode", x)[1]["W_launch"] == 1e-4
    with pytest.raises(ValueError, match="pipeline stage"):
        HostDeviceRecurrence(laws, (), lambda: 0.0, stage=(0, 2), paths=paths)
    bare = HostDeviceRecurrence(laws, (), lambda: 0.0)
    assert bare.path(BatchView((RequestShape(6, 9, True),), capture_rung=1)) is None


def test_a_run_file_s_host_entry_takes_paths_on_segments_its_laws_state():
    laws = {"decode": {"W_launch": [1e-4, 0, 0]}}
    compass_run._check_host(
        "run.json", {"laws": laws, PATHS: {"decode": {"W_launch": {"rung1": 2e-3}}}}
    )
    for bad in (
        {"decode": {"W_pre": {"rung1": 2e-3}}},
        {"decode": {"W_launch": {"rung1": -1.0}}},
        {"prefill": {"W_launch": {"eager": 1.0}}},
    ):
        with pytest.raises(ValueError, match=PATHS):
            compass_run._check_host("run.json", {"laws": laws, PATHS: bad})


def test_a_run_prices_a_prefill_at_its_path_s_constant(tmp_path, monkeypatch):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = {
        **compass_run.host_key(r.config),
        **keyed_platform(Kind.HOST_LAWS, PLATFORM),
    }
    key["engine"] = {**key["engine"], GRAPHS: SETS}
    laws = {k: {g: [1e-4, 0, 0] for g in HOST_SEGMENTS} for k in FORWARD_KINDS}
    paths = {"prefill": {"W_launch": {"eager": EAGER_S}}}
    run_file(tmp_path, monkeypatch, host={"key": key, "laws": laws, PATHS: paths})
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE,
        compass_run.channel_table(json.loads((tmp_path / "run.json").read_text())),
        None,
    )
    compass_run.runner(r, META_8B)
    batch, _ = scheduler_of([1000] * 5).schedule()
    assert batch.total_tokens_num > BUCKET
    r.forward(batch)
    assert batch.compass_path == "eager"
    assert r.compass_host.segments["W_launch"] == pytest.approx(EAGER_S)


def test_the_recurrence_grade_prices_a_path_constant_on_each_step(tmp_path):
    """`record()`'s output prefill of 50 tokens runs past a bucket of 32: its
    eager path's launch constant, 2 ms over its law's, moves its exit 2 ms."""
    steps = recurrence_record()
    samples, law = _labelled(tmp_path, steps), _law()
    bare = recurrence.priced(steps, samples, law)
    law["host"]["key"] = {"engine": {GRAPHS: {**SETS, "token": {"prefill": [32]}}}}
    base = law["host"]["laws"]["prefill"]["W_launch"][0]
    law["host"][PATHS] = {"prefill": {"W_launch": {"eager": base + 2e-3}}}
    priced = recurrence.priced(steps, samples, law)

    def residual(report):
        return {r[0]: r[3] for r in report["failures"]}

    assert residual(priced)[2] == pytest.approx(residual(bare).get(2, 0) + 2.0)
