# SPDX-License-Identifier: MIT
"""`compass fit steps`, run through `cli.main`.

Each record directory holds one of `test_backend_step_linear_fit.py`'s planted
timed records as `steps/dp0.jsonl`, its host key and engine file beside it,
or a data-parallel group's ranks as `test_step_linear_moe_column.py` plants
them, and a run.json written by hand as `test_fit_samples.py` writes one. The store
is a fresh directory; the entry's provenance names planted trees, since git
cannot name a read-only checkout in every container.
"""

import hashlib
import json
import re

import pytest
from test_artifact_store import AITER_ROOT, ATOM_ROOT
from test_backend_step_linear_fit import _steps, _timed_record, platform_arg
from test_fit_samples import DEVICE, SOURCE, edit, idle_rank
from test_memory_readings import DOCUMENT, STACK
from test_parity_served import SAMPLING
from test_step_linear_moe_column import (
    DP,
    EXPERTS,
    FULL,
    ROUTING,
    RUNG,
    SPARSE,
    _groups,
    _records,
)

from atom.compass import cli
from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    Key,
    Kind,
    Reading,
    Row,
    keyed_platform,
    machine_platform,
)
from atom.compass.backends import StepLinearLaw, step_linear_fit
from atom.compass.backends.step_labels import digest
from atom.compass.drive.aiter_jit import BUILDS
from atom.compass.fit import publish
from atom.compass.parity import TRANSFERS
from atom.compass.run.step_grid import BOUNDS
from atom.compass.spec import MachineSpec

ENGINE = {
    "ladder": [],
    "kv_cache_block_size": 16,
    "kv_cache_dtype": "bf16",
    "quantization": "unquantized-bf16",
    "tensor_parallel_size": 1,
    "data_parallel_size": 1,
}
HOST = {
    "host_cpu": "planted cpu",
    "atom": "0" * 64,
    "model": "sha256:" + "1" * 64,
    "engine": ENGINE,
}
LINEAR = ("--backend", "step-linear")
#: The platform of the run files' machine spec, the one every fit here names.
PLATFORM = machine_platform(MachineSpec.from_mapping(DOCUMENT))


@pytest.fixture(autouse=True)
def planted_roots(monkeypatch):
    monkeypatch.setattr(publish, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT))


def record(directory, seed, key=HOST, keyed=True, stack=STACK, trace="trace"):
    """`directory`, a record of `_steps(seed)` (`as_record`); without `keyed`,
    no host key beside it."""
    steps = directory / "steps"
    steps.mkdir(parents=True)
    _timed_record(steps / "dp0.jsonl", _steps(seed), key=key if keyed else None)
    return as_record(directory, key, stack, trace)


def as_record(directory, key=HOST, stack=STACK, trace="trace"):
    """`directory`, whose `steps/` holds each rank's lines, as a record under the
    host key `key`, each rank's engine file its engine. Every request is named,
    and `directory/client.json` is the export of one client run sending them
    all, each from trace `<trace>.<k>`, `k` its first step's index over 100."""
    engine = {**key["engine"], **dict.fromkeys(BOUNDS, 4096), "num_kvcache_blocks": 900}
    sent = {}
    for path in sorted((directory / "steps").glob("dp*[0-9].jsonl")):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for i, line in enumerate(lines):
            line["named"] = [[r[0], f"r{r[0]}"] for r in line["batch"]]
            for r in line["batch"]:
                at = {"trace": f"{trace}.{i // 100}", "sampling": SAMPLING}
                sent.setdefault(f"r{r[0]}", at)
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        path.with_suffix(".engine.json").write_text(json.dumps(engine))
    (directory / "client.json").write_text(json.dumps({"requests": sent}))
    run = {
        "format_version": 2,
        "argv": ["compass", "record", "--out", str(directory)],
        "source": SOURCE,
        "host_key": key,
        "device": [DEVICE],
        "stack": stack,
        "ptl": "disabled",
        "profiled": False,
    }
    (directory / "run.json").write_text(json.dumps(run))
    return directory


def fit(tmp_path, command, *args):
    """`compass <command> <args>`, publishing to `tmp_path/store`."""
    store, out = (str(tmp_path / f) for f in ("store", "out"))
    tail = ["--store", store, "--out", out]
    cli.main([*command.split(), *map(str, args), *tail])


def law_key(form="history", engine=ENGINE) -> Key:
    return Key.of(
        Kind.STEP_LINEAR_LAW,
        model=HOST["model"],
        engine_config=json.dumps(engine, sort_keys=True),
        source_root="atom_py-" + HOST["atom"],
        form=form,
        **keyed_platform(Kind.STEP_LINEAR_LAW, PLATFORM),
    )


def member(store, key, stem):
    return json.loads(
        ArtifactStore(store).read(key).read_member(stem, publish.RANK, "json")
    )


@pytest.mark.parametrize("form", step_linear_fit.FORMS)
def test_the_law_published_is_the_one_step_linear_fits_command_writes_less_its_host_laws(
    tmp_path, form
):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    fit(tmp_path, "fit steps", *LINEAR, "--form", form, a, b, *clients(a, b))

    law = member(tmp_path / "store", law_key(form), "params")
    # The command names runs in argument order, the samples in digest order.
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    law_file = str(tmp_path / "law.json")
    step_linear_fit.main(
        [*map(str, paths), "--form", form, *platform_arg(DOCUMENT)]
        + ["--out", law_file]
    )
    written = json.loads((tmp_path / "law.json").read_text())
    assert law == {k: v for k, v in written.items() if k not in ("host", "host_fit")}
    StepLinearLaw(law)
    copy = json.loads((tmp_path / "out/step_linear_law.json").read_text())
    report = member(tmp_path / "store", law_key(form), "report")
    assert copy == {
        "key": law_key(form).values,
        "token": law_key(form).token(),
        "params": law,
        "report": report,
    }


def test_a_routed_law_is_the_one_step_linear_fits_command_writes_with_the_routing(
    tmp_path,
):
    # Two runs of a group of `DP` ranks priced by a law whose rung adopts the
    # experts column: one run's ranks all full, the other's mostly idle.
    key = {**HOST, "engine": {**ENGINE, "data_parallel_size": DP}}
    runs = []
    for name, seed, others in (("full", 0, FULL), ("sparse", 1, SPARSE)):
        (tmp_path / name).mkdir()
        _records(tmp_path / name / "steps", _groups(seed, 200, others))
        runs.append(as_record(tmp_path / name, key))
    routing = ("--moe-routing", ",".join(map(str, ROUTING)))
    fit(tmp_path, "fit steps", *LINEAR, *routing, *runs, *clients(*runs))

    law = member(tmp_path / "store", law_key(engine=key["engine"]), "params")
    # The samples come in run order, a run named by its ranks' digests.
    ranks = sorted(
        (sorted((d / "steps").glob("dp*[0-9].jsonl")) for d in runs),
        key=lambda files: ", ".join(map(digest, files)),
    )
    law_file = str(tmp_path / "law.json")
    step_linear_fit.main(
        [str(f) for files in ranks for f in files]
        + ["--form", "history", *routing, *platform_arg(DOCUMENT), "--out", law_file]
    )
    written = json.loads((tmp_path / "law.json").read_text())
    assert law == {k: v for k, v in written.items() if k not in ("host", "host_fit")}
    assert law["moe_routing"] == {"experts": ROUTING[0], "top_k": ROUTING[1]}
    assert list(law["decode"][str(RUNG)]["coefficients"]) == list(EXPERTS)
    StepLinearLaw(law)


def test_step_linear_fits_the_history_form_unless_told_another(tmp_path):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    for name, *form in (("default",), ("base", "--form", "base")):
        (tmp_path / name).mkdir()
        fit(tmp_path / name, "fit steps", *LINEAR, *form, a, b, *clients(a, b))

    law = member(tmp_path / "default/store", law_key("history"), "params")
    assert law["form"] == "history"
    assert list(law["prefill"]["coefficients"]) == list(step_linear_fit.HISTORY)
    base = member(tmp_path / "base/store", law_key("base"), "params")
    assert list(base["prefill"]["coefficients"]) == list(step_linear_fit.PREFILL)


def test_fit_steps_twice_on_the_same_inputs_refuses_the_second_as_the_same_members(
    tmp_path,
):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    fit(tmp_path, "fit steps", *LINEAR, "--seed", 7, a, b, *clients(a, b))

    with pytest.raises(SystemExit, match="this publish holds the same members"):
        fit(tmp_path, "fit steps", *LINEAR, "--seed", 7, b, a, *clients(b, a))


def clients(*records):
    """`--client-run` with each record's export."""
    return [x for d in records for x in ("--client-run", d, d / "client.json")]


def served(key, traced) -> list[dict]:
    """The report's provenance of the records `traced` names, each with its trace."""
    named = {digest(d / "steps/dp0.jsonl"): t for d, t in traced.items()}
    return [
        {
            "source": "record",
            "run": run,
            "config": key,
            "traces": [f"{named[run]}.{k}" for k in range(3)],
            "profiled": False,
        }
        for run in sorted(named)
    ]


def test_the_report_names_each_run_with_every_trace_its_steps_served(tmp_path):
    a = record(tmp_path / "a", 19, trace="trace-a")
    b = record(tmp_path / "b", 20, trace="trace-b")
    fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))

    report = member(tmp_path / "store", law_key(), "report")
    assert report["provenance"] == served(HOST, {a: "trace-a", b: "trace-b"})


def test_the_entry_s_readings_are_the_ones_its_records_state(tmp_path):
    a = record(tmp_path / "a", 19)
    fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))

    entry = ArtifactStore(tmp_path / "store").read(law_key())
    assert dict(entry.fingerprints[Row.STEP_LINEAR_LAW].cells) == {
        Axis.SOFTWARE_STACK: Reading(
            "stated", "rocm 7.2.4 / aiter f4e7c7509 / rccl 2.22.3"
        ),
        Axis.TORCH: Reading("stated", "2.10.0"),
        Axis.ATOM_SRC: Reading("atom_py", HOST["atom"]),
        Axis.MODEL: Reading("stated", HOST["model"]),
        Axis.DEVICE: Reading("stated", "MI355X gfx950"),
        Axis.PTL: Reading("stated", "disabled"),
        Axis.ENGINE_CONFIG: Reading("stated", json.dumps(ENGINE, sort_keys=True)),
    }
    assert entry.provenance.source_roots == (ATOM_ROOT, AITER_ROOT)


def test_records_under_two_stacks_refuse_naming_the_component_and_both_values(
    tmp_path,
):
    a = record(tmp_path / "a", 19)
    b = record(tmp_path / "b", 20, stack={**STACK, "torch": "2.9.0"})

    named = f'stack torch: "2.10.0" in {a}; "2.9.0" in {b}'
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))


@pytest.mark.parametrize(
    "args, named",
    [
        (["--backend", "step-stub"], "stated, not fitted"),
        (
            ["--backend", "step-op"],
            "no backend step-op is fitted; step-linear and step-grid are",
        ),
        (
            ["--backend", "step-linear", "--grid-definition", "g.json"],
            "--grid-definition is read by step-grid, not step-linear",
        ),
        (
            ["--backend", "step-linear", "--grid=g"],
            "--grid is read by step-grid, not step-linear",
        ),
        (
            ["--backend", "step-linear", "0f1e2d3c4b5a"],
            "0f1e2d3c4b5a is not a record directory",
        ),
        (
            ["--backend", "step-linear", "--client-run", "c", "x.json"],
            "names c, which is not a RECORD",
        ),
    ],
)
def test_what_step_linear_does_not_fit_refuses_by_name(tmp_path, args, named):
    a = record(tmp_path / "a", 19)

    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *args, a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_a_record_no_client_run_names_refuses_naming_it(tmp_path):
    a = record(tmp_path / "a", 19)

    with pytest.raises(SystemExit, match=re.escape(f"{a} is named by no --client-run")):
        fit(tmp_path, "fit steps", *LINEAR, a)
    assert not (tmp_path / "store").exists()


def test_a_deployment_s_prefill_side_fits_from_the_deployment_s_export(tmp_path):
    # Prefill's record names each request `p-<key>`; the export, beside
    # prefill/ and decode/, names it `d-<key>`, as decode's response id that
    # decode's transfers join to prefill's DP rank and request id.
    p = record(tmp_path / "cell/prefill", 19)
    path = p / "steps/dp0.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line in lines:
        line["named"] = [[r, k, f"p-{k}"] for r, k in line["named"]]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    sent = json.loads((p / "client.json").read_text())["requests"]
    export = tmp_path / "cell/client_run.json"
    export.write_text(json.dumps({"requests": {f"d-{k}": v for k, v in sent.items()}}))
    joined = [
        {"response_id": f"d-{k}", "remote_dp_rank": 0, "transfer_id": int(k[1:])}
        for k in sent
    ]
    decode = tmp_path / "cell/decode/steps"
    decode.mkdir(parents=True)
    (decode / TRANSFERS).write_text("".join(json.dumps(t) + "\n" for t in joined))
    fit(tmp_path, "fit steps", *LINEAR, p, "--client-run", p, export)

    report = member(tmp_path / "store", law_key(), "report")
    assert report["provenance"] == served(HOST, {p: "trace"})


@pytest.mark.parametrize(
    "backend, named",
    [
        ("step-linear", "--sessions is read by step-grid, not step-linear"),
        ("step-grid", "{b} is named by no --client-run"),
    ],
)
def test_sessions_refuse_off_step_grid_and_an_unnamed_grid_input_refuses_under_them(
    tmp_path, backend, named
):
    a, b = record(tmp_path / "a", 19), tmp_path / "b"
    grid = ["--grid", b, "--grid-definition", "g"] if backend == "step-grid" else []
    sessions = ["--sessions", "0:4", *clients(a)]

    with pytest.raises(SystemExit, match=re.escape(named.format(b=b))):
        fit(tmp_path, "fit steps", "--backend", backend, a, *grid, *sessions)
    assert not (tmp_path / "store").exists()


def test_the_routing_refuses_off_step_linear(tmp_path):
    a = record(tmp_path / "a", 19)
    routed = ("--backend", "step-grid", "--moe-routing", "64,4")

    named = "--moe-routing is read by step-linear, not step-grid"
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *routed, a, *clients(a))
    assert not (tmp_path / "store").exists()


#: A MoE model's config.json, as DeepSeek's names its routed experts and top-k.
MOE = {"model_type": "deepseek_v3", "n_routed_experts": 64, "num_experts_per_tok": 4}


def kimi(**text) -> dict:
    """A Kimi-K3 config.json, whose text config states `text`, as Kimi's names
    its routed experts and top-k."""
    return {
        "model_type": "kimi_k3",
        "text_config": {"model_type": "kimi_linear", **text},
    }


def model(directory, config) -> tuple:
    """`directory`, a model directory whose config.json states `config`, and
    the host key of a record that served it."""
    directory.mkdir()
    text = json.dumps(config)
    (directory / "config.json").write_text(text)
    return directory, {
        **HOST,
        "model": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
    }


@pytest.mark.parametrize(
    "config, routing",
    [
        (MOE, (64, 4)),
        (
            {"model_type": "mixtral", "num_local_experts": 8, "num_experts_per_tok": 2},
            (8, 2),
        ),
        (
            {"model_type": "qwen3_moe", "num_experts": 128, "num_experts_per_tok": 8},
            (128, 8),
        ),
        (kimi(num_experts=32, num_experts_per_token=4), (32, 4)),
        (
            {
                "model_type": "minimax_m3_vl",
                "text_config": {"num_local_experts": 16, "num_experts_per_tok": 4},
            },
            (16, 4),
        ),
        ({"model_type": "llama"}, None),
    ],
)
def test_the_law_states_the_routing_the_served_model_s_config_names(
    tmp_path, config, routing
):
    m, key = model(tmp_path / "m", config)
    a = record(tmp_path / "a", 19, key=key)
    fit(tmp_path, "fit steps", *LINEAR, "--model", m, a, *clients(a))

    law = json.loads((tmp_path / "out/step_linear_law.json").read_text())["params"]
    stated = routing and {"experts": routing[0], "top_k": routing[1]}
    assert law.get("moe_routing") == stated


@pytest.mark.parametrize(
    "served, config, named",
    [
        (
            HOST,
            MOE,
            "--model {m} holds config.json {digest}; the records served {served}",
        ),
        (
            None,
            kimi(num_experts=32),
            "--model {m} names 32 routed experts and no top-k",
        ),
    ],
)
def test_a_model_the_records_did_not_serve_or_with_no_top_k_refuses(
    tmp_path, served, config, named
):
    m, key = model(tmp_path / "m", config)
    a = record(tmp_path / "a", 19, key=served or key)

    named = named.format(m=m, digest=key["model"], served=HOST["model"])
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *LINEAR, "--model", m, a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_the_model_refuses_beside_the_routing_and_off_step_linear(tmp_path, capsys):
    m, key = model(tmp_path / "m", MOE)
    a = record(tmp_path / "a", 19, key=key)

    with pytest.raises(SystemExit):
        both = ("--moe-routing", "64,4", "--model", m)
        fit(tmp_path, "fit steps", *LINEAR, *both, a, *clients(a))
    assert "--model: not allowed with argument --moe-routing" in capsys.readouterr().err
    grid = ("--backend", "step-grid", "--model", m)
    named = "--model is read by step-linear, not step-grid"
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *grid, a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_the_join_width_is_the_engine_configs_not_the_record_files(tmp_path):
    a = record(
        tmp_path / "a", 19, key={**HOST, "engine": {**ENGINE, "data_parallel_size": 2}}
    )

    with pytest.raises(SystemExit, match="are not runs of 2 ranks"):
        fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))


def test_a_dp_record_a_rank_of_which_gave_no_sample_refuses_at_its_width(tmp_path):
    a = record(
        tmp_path / "a", 19, key={**HOST, "engine": {**ENGINE, "data_parallel_size": 2}}
    )
    idle_rank(a / "steps")

    with pytest.raises(SystemExit, match="are not runs of 2 ranks"):
        fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_records_predating_the_data_parallel_width_refuse_naming_it(tmp_path):
    engine = {k: v for k, v in ENGINE.items() if k != "data_parallel_size"}
    a = record(tmp_path / "a", 19, key={**HOST, "engine": engine})
    with pytest.raises(SystemExit, match="states no data_parallel_size"):
        fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))


def test_a_prefill_context_parallel_record_refuses_naming_its_width(tmp_path):
    engine = {**ENGINE, "prefill_context_parallel_size": 2}
    a = record(tmp_path / "a", 19, key={**HOST, "engine": engine})
    with pytest.raises(SystemExit, match="states prefill_context_parallel_size 2"):
        fit(tmp_path, "fit steps", *LINEAR, a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_a_record_whose_server_built_an_aiter_module_refuses_naming_it(tmp_path):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    edit(b / "run.json", lambda r: r.update({BUILDS: ["mha_varlen_fwd"]}))

    named = (
        f"{b} is not fitted: its server ran a JIT compile of ['mha_varlen_fwd'] while"
    )
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))
    assert not (tmp_path / "store").exists()
