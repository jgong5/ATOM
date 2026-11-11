# SPDX-License-Identifier: MIT
"""A model's launch features as host-law columns, on a planted provider.

`Planted` says a prefill past its runner's largest recorded prefill bucket
launches eagerly, 40 ms dearer. A recording runner writes what it observed on
each line and the graph sets in its key, which labelling and the fit's main
keep; `fit_host` refuses a step it predicts otherwise and adopts the column
where the held-out error falls; a run's forward and the recurrence grade price
it from `predict`; the runner, the server and the run file pick a host entry by
its key less the sets. With no provider, nothing moves.
"""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_grid import scheduler_of
from test_backend_step_linear_fit import SPEC, _steps, _timed_record
from test_fit_steps import DIGEST
from test_memory_readings import META_8B
from test_parity_device_window import _recorded
from test_parity_record import _config, _drive
from test_parity_recurrence import _labelled, _law, record
from test_run_hops import model, parsed
from test_run_step_grid import deployment, publish, run_file
from test_served_tokenizer import (
    PROMPT_TOKENS,
    TERM,
    _charges,
    _run_file,
    _Tokenizer,
    rt,  # noqa: F401  the rt fixture
)

from atom.compass import run as compass_run
from atom.compass.backends.launch_features import (
    GRAPHS,
    REGISTRY,
    LaunchFeatures,
    stated,
)
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import BatchView, RequestShape, sum_tokens
from atom.compass.backends.step_labels import Sample
from atom.compass.backends.step_linear_fit import (
    fit_host,
    host_held_out,
    label_records,
    main,
)
from atom.compass.drive import runfile
from atom.compass.parity import ENV, compare, read, recurrence
from atom.compass.run.step_grid import keyed
from atom.compass.runner.overrides import (
    FORWARD_KINDS,
    HOST_SEGMENTS,
    HostDeviceRecurrence,
)

ARCH = "PlantedForCausalLM"
#: A bucket past which a third of `_samples`' prefills launch eagerly: a
#: minority, which a `W_launch` fit blind to the column cuts as compiles.
BUCKET = 4000
SETS = {"architecture": ARCH, "token": {"prefill": [BUCKET], "decode": []}}
EAGER_S = 0.04


class Planted(LaunchFeatures):
    features = ("eager",)

    def install(self, runner):
        runner.launches = 0

    def observe(self, runner):
        runner.launches += 1
        return {"eager": float(runner.launches % 2)}

    def predict(self, view, graphs):
        bucket = max(graphs["token"]["prefill"])
        return {"eager": float(sum_tokens(view.requests) > bucket)}


@pytest.fixture(autouse=True)
def planted(monkeypatch):
    monkeypatch.setitem(REGISTRY, ARCH, Planted())


def _samples(graphs=SETS, seed=3068, runs=("a", "b")):
    """`runs` of output prefills and decodes; a prefill's `W_launch` is 2 ms
    plus 1 us a token, and `EAGER_S` more past `BUCKET` tokens."""
    rng, out = np.random.default_rng(seed), []
    key = {"engine": {GRAPHS: graphs} if graphs else {}}
    for run in runs:
        for i in range(60):
            tokens = int(rng.integers(256, 6000)) if i % 2 else 1
            view = BatchView(
                (RequestShape(tokens, tokens, i % 2 == 0),), produces_output=True
            )
            launch = Planted().predict(view, graphs) if graphs else {}
            w = 2e-3 + 1e-6 * tokens + EAGER_S * launch.get("eager", 0.0)
            host = {"E": 1e-4, "W_pre": 1e-4, "W_post": 1e-4}
            host["W_launch"] = w * rng.uniform(0.99, 1.01)
            provenance = {
                "source": "record",
                "run": run,
                "requests": [i],
                "config": key,
            }
            out.append(
                Sample(i, 0, view, 0.01, None, host, False, provenance, launch=launch)
            )
    return out


def test_a_planted_column_is_adopted_where_it_lowers_the_held_out_error_and_priced():
    host, report = fit_host(_samples())
    # Solved with the column, no eager launch is cut as a compile.
    assert report["jit_steps"] == {}
    column = host["columns"]["prefill"]["W_launch"]
    assert column == pytest.approx({"eager": EAGER_S}, rel=0.05)
    assert host["laws"]["prefill"]["W_launch"] == pytest.approx(
        [2e-3, 1e-6, 0], abs=2e-4
    )
    adopted = report["prefill.W_launch"]["launch_columns"]
    assert adopted["adopted"] == ["eager"] and adopted["held_out"][1]["eager"] < 0.01
    # A decode never launches eagerly: its column has no variance to adopt.
    assert report["decode.W_launch"]["launch_columns"]["adopted"] == []

    rec = HostDeviceRecurrence(
        host["laws"],
        (),
        lambda: 0.0,
        columns=host["columns"],
        launch=stated(host["key"]),
    )
    for tokens, eager in ((1000, 0.0), (BUCKET + 1000, EAGER_S)):
        view = BatchView((RequestShape(tokens, tokens, False),), produces_output=True)
        batch = _batch(tokens, compass_launch=rec.predicted(view))
        rec.charge(batch, 0.01, 0.0)
        law = np.dot(host["laws"]["prefill"]["W_launch"], (1, tokens, 1))
        assert rec.segments["W_launch"] == pytest.approx(law + eager, rel=0.05)
    # Each of three runs held out, the other two adopt the column and price
    # its eager prefills with it.
    held = host_held_out(_samples(runs=("a", "b", "c")))["host prefill W_launch"]
    assert np.abs(held).max() < 0.05


def _batch(tokens, **fields):
    return SimpleNamespace(
        total_seqs_num_prefill=1,
        total_tokens_num=tokens,
        total_seqs_num=1,
        produces_output=lambda: True,
        **fields,
    )


def test_a_step_predicted_otherwise_than_it_launched_refuses_by_name():
    samples = _samples()
    flipped = {"eager": 1 - samples[7].launch["eager"]}
    samples[7] = dataclasses.replace(samples[7], launch=flipped)
    with pytest.raises(ValueError, match="a step 7 launched"):
        fit_host(samples)


def test_without_a_provider_the_fit_adopts_and_reports_nothing_new():
    host, report = fit_host(_samples(graphs=None))
    assert "columns" not in host and GRAPHS not in host["key"]["engine"]
    assert not any(
        "launch_columns" in v for v in report.values() if isinstance(v, dict)
    )


def test_a_column_is_refused_where_it_cannot_be_priced():
    columns = {"prefill": {"W_launch": {"eager": EAGER_S}}}
    with pytest.raises(ValueError, match=r"\['eager'\], which .* LaunchFeatures"):
        HostDeviceRecurrence({}, (), lambda: 0.0, columns=columns, launch=stated({}))
    laws = {"prefill": {g: [1e-4, 0, 0] for g in HOST_SEGMENTS}}
    launch = stated({"engine": {GRAPHS: SETS}})
    rec = HostDeviceRecurrence(laws, (), lambda: 0.0, columns=columns, launch=launch)
    with pytest.raises(CostRefused, match=r"no law for prefill\.W_launch\.eager"):
        rec.charge(_batch(4000), 0.01, 0.0)
    with pytest.raises(ValueError, match="pipeline stage"):
        HostDeviceRecurrence(
            laws, (), lambda: 0.0, (0, 2), columns=columns, launch=launch
        )


@pytest.mark.parametrize("architectures", [[ARCH], ["OtherForCausalLM"]])
def test_a_recording_runner_writes_what_it_observed_and_the_sets_in_its_key(
    tmp_path, monkeypatch, architectures
):
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1])
    config = _config(hf_config=type("Hf", (), {"architectures": architectures}))
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = recorded(0, config)
    runner.host_key = {"engine": {"ladder": [1]}}
    for batch in batches:
        runner.forward(batch)
    device.tick = 10**9  # every device event completes: `exit` writes every line
    runner.exit()
    lines = read(tmp_path / "real")[0]
    key = json.loads((tmp_path / "real" / "dp0.key.json").read_text())
    samples = label_records([tmp_path / "real" / "dp0.jsonl"])[0]
    if architectures == [ARCH]:
        assert [s["launch"] for s in lines] == [
            {"eager": (n + 1) % 2} for n in range(len(lines))
        ]
        assert key["engine"][GRAPHS]["architecture"] == ARCH
        assert set(key["engine"][GRAPHS]) == {"architecture", "token", "request"}
        # Each labelled sample carries its line's `launch` to the fit.
        assert {s.step: s.launch for s in samples} == {
            s.step: lines[s.step]["launch"] for s in samples
        }
        assert {s.launch["eager"] for s in samples} == {0, 1}
    else:
        assert not any("launch" in s for s in lines) and key == runner.host_key
        assert samples and not any(s.launch for s in samples)
    # A real line's `launch` is no part of a step's decision.
    first = compare(tmp_path / "real", tmp_path / "simulated")["first_divergence"]
    assert first == {0: None}


def test_the_fit_s_main_keeps_its_records_graph_sets_in_the_host_key(tmp_path):
    """A timed record whose lines carry `Planted`'s launch and whose key states
    `SETS`: the law's host key keeps the sets, which its engine config, read
    from the engine file, does not state, and `W_launch` is offered the
    column."""
    path, _ = _timed_record(tmp_path / "dp0.jsonl", _steps(14, n=400))
    key = json.loads(path.with_suffix(".key.json").read_text())
    key["engine"][GRAPHS] = SETS
    path.with_suffix(".key.json").write_text(json.dumps(key))
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line in lines:
        line["launch"] = {"eager": float(sum(r[0] for r in line["rows"]) > BUCKET)}
    assert {line["launch"]["eager"] for line in lines} == {0, 1}
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))

    main([str(path), "--out", str(tmp_path / "law.json"), *SPEC])
    law = json.loads((tmp_path / "law.json").read_text())
    assert law["host"]["key"]["engine"][GRAPHS] == SETS
    assert GRAPHS not in law["engine_config"]
    assert "launch_columns" in law["host_fit"]["prefill.W_launch"]


def test_a_run_prices_a_column_under_the_sets_its_host_laws_key_states(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = {**compass_run.host_key(r.config), "machine_spec": DIGEST}
    key["engine"] = {**key["engine"], GRAPHS: SETS}
    laws = {k: {g: [1e-4, 0, 0] for g in HOST_SEGMENTS} for k in FORWARD_KINDS}
    # An entry fitted under another engine config, which the run passes over.
    elsewhere = {**key, "engine": {**key["engine"], "kv_cache_dtype": "fp8"}}
    for feature in ("eager", "other"):
        columns = {"prefill": {"W_launch": {feature: EAGER_S}}}
        entry = {"key": key, "laws": laws, "columns": columns}
        run_file(tmp_path, monkeypatch, host=[{**entry, "key": elsewhere}, entry])
        monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
        compass_run._PublishedClock(
            compass_run.ENGINE,
            compass_run.channel_table(json.loads((tmp_path / "run.json").read_text())),
            None,
        )
        if feature == "other":
            with pytest.raises(ValueError, match="'other'"):
                compass_run.runner(r, META_8B)
            continue
        compass_run.runner(r, META_8B)
        batch, _ = scheduler_of([1000] * 5).schedule()
        assert batch.total_tokens_num > BUCKET
        r.forward(batch)
        assert r.compass_host.segments["W_launch"] == pytest.approx(1e-4 + EAGER_S)


def test_the_server_and_the_run_file_take_the_host_entry_the_runner_takes(
    tmp_path, monkeypatch, rt  # noqa: F811
):
    """Two host entries, each fitted under its own engine config by a runner
    that stated its graph sets in the key, which a simulated run's config never
    states: the run file's hops and the server's tokenizer and intake terms
    come from the entry the runner takes, the one fitted under this run's."""
    flags = []
    for i, block in enumerate((16, 32)):
        r = deployment(tmp_path, kv_cache_block_size=block)
        encode = {"fixed_s": i + 1.0, "tokens_per_s": 1e6}
        host = {
            "key": {"engine": {**keyed(r.config), GRAPHS: SETS}},
            "hops": {"ipc_s": (i + 1) * 1e-4},
            "tokenizer": TERM | {"encode": encode},
            "http_intake": {"fixed_s": (i + 1) * 1e-3, "per_byte_s": 0.0},
        }
        (tmp_path / f"law{i}.json").write_text(json.dumps({"host": host}))
        flags += ["--host-laws", str(tmp_path / f"law{i}.json")]
    # This run's ATOM arguments resolve the second deployment's config.
    monkeypatch.setattr("atom.compass.drive.run._config", lambda atom_args: r.config)
    run = runfile.build(parsed(tmp_path, *flags), model(tmp_path))
    _run_file(monkeypatch, tmp_path, run)
    monkeypatch.setattr(compass_run, "_frontend_host", None)
    tok = _Tokenizer()
    compass_run.tokenizer(tok, r.config)
    taken = {**run}
    compass_run.pick_fitted(taken, "host", r.config)
    entry = taken["host"]

    assert entry == run["host"][1]
    assert run["ipc_s"] == entry["hops"]["ipc_s"]
    assert compass_run.http_intake()[0] == entry["http_intake"]["fixed_s"]
    encode = entry["tokenizer"]["encode"]
    assert _charges(tok, rt)[0] == pytest.approx(
        encode["fixed_s"] + PROMPT_TOKENS / encode["tokens_per_s"]
    )


def test_the_recurrence_grade_prices_a_column_on_each_step_it_predicts(tmp_path):
    """`record()`'s output prefill of 50 tokens runs past a bucket of 32: its
    launch, and so its exit, move 2 ms later."""
    steps = record()
    samples, law = _labelled(tmp_path, steps), _law()
    bare = recurrence.priced(steps, samples, law)
    law["host"]["key"] = {"engine": {GRAPHS: {**SETS, "token": {"prefill": [32]}}}}
    law["host"]["columns"] = {"prefill": {"W_launch": {"eager": 2e-3}}}
    priced = recurrence.priced(steps, samples, law)

    def residual(report):
        return {r[0]: r[3] for r in report["failures"]}

    assert residual(priced)[2] == pytest.approx(residual(bare).get(2, 0) + 2.0)
