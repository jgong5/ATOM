# SPDX-License-Identifier: MIT
"""`compass advice SPACE OUT`'s reuse check: each point's laws and memory
readings get a verdict naming the requirement that failed, written into OUT.

The configuration is `test_run_step_grid`'s deployment, resolved for every
instance, with its readings and platform readings in the store; its laws are
the checked-in DPA8 decode law re-keyed to that configuration, as a law file
(`reuse.LAWS`) or a store entry beside host laws. The 1P1D sides whose config
resolves for real are DeepSeek-V4-Pro's, judged by the shipped crs005 DPA8
pair. A pipeline's law is that law with a part per stage, served to the
configuration on pipeline stages as a producer. The head is the tree's own
digest, except where a test builds a repository for the tree.
"""

import functools
import json
import os
import pickle
import shutil
import subprocess

import platforms
import pytest
from test_artifact_invalidation import STANZA, moved
from test_capture_v4_pro import CONFIG as V4_PRO
from test_drive_runfile import LAWS as HOST_LAWS
from test_law_file_readings import law_member
from test_law_token_graph_buckets import RECORDED
from test_memory_readings import READINGS, publish_readings
from test_page_bound_reading import V41
from test_run_check_side import (
    DECODE_SIZES,
    MACHINE,
    PAIR,
    SHARED_ENV,
    SIDES,
    UNCOVERED,
)
from test_run_lacking import served

#: The requirement each side's law in the crs005 pair is calibrated for at its
#: own key: the decode side's coverage, the prefill side's accuracy.
COVERAGE_REQUIREMENT = {"prefill": 4, "decode": 3}
from test_run_lacking_rungs import DPA8
from test_run_step_grid import DOCUMENT, PLATFORM, deployment, host_laws_key, readings

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import cli
from atom.compass import run as compass_run
from atom.compass.advice import check
from atom.compass.advice.check import REUSE, STATES
from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    GateState,
    Key,
    Kind,
    Topology,
    keyed_platform,
    member_name,
)
from atom.compass.backends.json_file import read_json
from atom.compass.bench import ops
from atom.compass.drive import reuse
from atom.compass.drive import run as drive_run
from atom.compass.fit.publish import RANK
from atom.compass.memory.readings import PAGE_BOUND
from atom.compass.run import memory
from atom.compass.run.step_grid import keyed

#: Rungs 1, 2, 4 and 8 captured within a bound of 8, which the DPA8 law covers.
COVERED = {"capture_sizes": [1, 2, 4, 8, 16], "max_num_seqs": 8}
WITHIN = {"steps": 9, "on_totals": 0.01, "held_out_step": 3, "held_out": 0.01}
WITHIN |= {"median": -0.02, "refused": 0}


@pytest.fixture
def at_head(monkeypatch):
    """The head taken, a commit whose ATOM digest is the tree's."""
    head = {"commit": "c0ffee" * 6 + "c0ff", "ref": "origin/" + reuse.INTEGRATION}
    head["atom"] = compass_run.atom_digest()
    monkeypatch.setattr(reuse, "head", lambda commit=None: head)


@pytest.fixture
def searched(tmp_path, monkeypatch, at_head):
    """The deployment at `COVERED`, served for every instance, at the head
    whose digest is the tree's, with no law file; each start the check makes
    answers, so a point it would start is searchable (`test_advice_check`
    starts one)."""
    r = deployment(tmp_path, **COVERED)
    served(monkeypatch, r.config)
    monkeypatch.setattr(check, "_start", lambda *args: None)
    resolved_here(monkeypatch, lambda atom_args: r.config)
    monkeypatch.setattr(reuse, "LAWS", tmp_path / "laws")
    (tmp_path / "laws").mkdir()
    return r


#: Each config `check._resolved` resolved in this module, by its arguments,
#: arch and this process's environment, and each law file `reuse.law` read, by
#: path; pickled.
RESOLVED, READ = {}, {}
DIGEST = compass_run.atom_digest


@functools.cache
def tree_digest() -> str:
    """This tree's ATOM digest, taken at the first call."""
    return DIGEST()


@pytest.fixture(autouse=True)
def digested_once(monkeypatch):
    """`atom_digest` of this tree taken once for the module, as no case edits
    the tree; of any other root, at each call."""
    monkeypatch.setattr(
        compass_run,
        "atom_digest",
        lambda root=None: tree_digest() if root is None else DIGEST(root),
    )


@pytest.fixture(scope="module")
def v4_pro(tmp_path_factory):
    """DeepSeek-V4-Pro's model directory, one for the module, so each 1P1D
    case serves its sides at the same arguments."""
    model = tmp_path_factory.mktemp("models") / "DeepSeek-V4-Pro"
    model.mkdir()
    shutil.copy(V4_PRO, model / "config.json")
    return model


@pytest.fixture
def resolved_once(monkeypatch):
    """Each instance's config resolved in the check's child once per module at
    its arguments, arch and environment, as alike ones resolve alike; and each
    shipped law file read once, as no case edits one."""
    resolve, read = check._resolved, reuse.read_json

    def read_once(path):
        if path not in READ:
            READ[path] = pickle.dumps(read(path))
        return pickle.loads(READ[path])

    def once(args, arch):
        env = sorted(
            (k, v) for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"
        )
        at = (tuple(args), arch, tuple(env))
        if at not in RESOLVED:
            RESOLVED[at] = pickle.dumps(resolve(args, arch))
        return pickle.loads(RESOLVED[at])

    monkeypatch.setattr(check, "_resolved", once)
    monkeypatch.setattr(reuse, "read_json", read_once)


def resolved_here(monkeypatch, resolve):
    """Each instance's config `resolve(atom_args)`, resolved in this process
    rather than in the check's child; any other child stays one."""
    child = ops._child

    def here(fn, *args):
        return fn(*args) if args[:1] == (drive_run._declared,) else child(fn, *args)

    monkeypatch.setattr(ops, "_child", here)
    monkeypatch.setattr(
        drive_run, "_declared", lambda atom_args, arch: resolve(atom_args)
    )


def advise(tmp_path, document=DOCUMENT, **fields) -> dict:
    """`compass advice` on one candidate at concurrencies 1 and 4, serving
    the model in `tmp_path` on `document`'s platform, the suite's, from its
    store, with `fields` in place of the space's own; the verdicts it
    writes."""
    flags = platforms.flags(tmp_path / "store", document, tmp_path)
    space = {
        "model": str(tmp_path),
        "machine": {"device": flags[1], "ptl": flags[3]},
        "store": str(tmp_path / "store"),
        "workload": {"traces": "/t.jsonl", "select": "agentx", "concurrency": [1, 4]},
        "slo": {},
        "common": {"basis": "a recipe"},
        "dimensions": {"config": {"a": {}}},
        **fields,
    }
    (tmp_path / "space.json").write_text(json.dumps(space))
    cli.main(["advice", str(tmp_path / "space.json"), str(tmp_path / "out")])
    return json.loads((tmp_path / "out" / REUSE).read_text())


def verdicts(judged: dict, what: str, side: str = "server") -> list[dict]:
    """The one candidate's `what` verdicts on its `side`, a point's each."""
    points = judged["candidates"]["a"]["points"]
    return [point[what][side] for point in points.values()]


def law(r, **changed) -> dict:
    """The DPA8 decode law re-keyed to `r`'s config and platform at this
    tree's ATOM digest, each source's figures within the gate, with
    `changed`."""
    held = read_json(DPA8)
    held["engine_config"] = keyed(r.config)
    held["platform"] = keyed_platform(Kind.STEP_LINEAR_LAW, PLATFORM)
    held["host"]["key"] = compass_run.host_key(r.config)
    held["host"]["key"] |= keyed_platform(Kind.HOST_LAWS, PLATFORM)
    held["records_atom"] = held["host"]["key"]["atom"]
    for source in held["sources"].values():
        source["figures"] = {run: {"decode": dict(WITHIN)} for run in held["runs"]}
    return held | changed


def past(r) -> tuple[dict, str]:
    """`law` with one run's decode median +5.12% on source refitted, past the
    gate; and that run."""
    held = law(r)
    run = held["runs"][1]
    held["sources"]["refitted"]["figures"][run]["decode"]["median"] = 0.0512
    return held, run


def law_file(tmp_path, held: dict) -> str:
    path = tmp_path / "laws" / "law.json"
    path.write_text(json.dumps(held))
    return str(path)


def publish(tmp_path, r, kind, key, params, conditions=None, store="store"):
    """`params` handed off at `key` in the store at `tmp_path / store`, under
    `conditions`, else `r`'s readings."""
    return ArtifactStore(tmp_path / store).publish(
        Key.of(kind, **key),
        provenance=STANZA,
        topology=Topology(),
        conditions=conditions or readings(r),
        gates=GateState.of(),
        members={member_name("params", RANK, "json"): json.dumps(params).encode()},
    )


def stored(
    tmp_path, r, conditions=None, atom=None, host=True, store="store", held=None
):
    """`held`'s step law, else `law`'s, in the store at `tmp_path / store`, its
    records at `atom`, else at the head's digest, under `conditions`; with
    `host`, its host laws beside it, keyed to the CPU `held` states."""
    held = held or law(r)
    key = {
        "model": compass_run._file_digest(r.config.model, "config.json"),
        "engine_config": json.dumps(keyed(r.config), sort_keys=True),
        "source_root": compass_run.source_root(atom),
        "form": held["form"],
        **keyed_platform(Kind.STEP_LINEAR_LAW, PLATFORM),
    }
    step = {k: v for k, v in held.items() if k not in ("host", "host_fit")}
    entry = publish(tmp_path, r, Kind.STEP_LINEAR_LAW, key, step, conditions, store)
    if host:
        params = {"key": held["host"]["key"], "laws": HOST_LAWS}
        key = host_laws_key(r) | {"host_cpu": held["host"]["key"]["host_cpu"]}
        publish(tmp_path, r, Kind.HOST_LAWS, key, params, None, store)
    return f"{entry.key.kind} {entry.key.token()} in {tmp_path / store}"


def test_a_law_and_readings_at_the_head_s_key_are_reused_at_every_point(
    tmp_path, searched, capsys
):
    where = law_file(tmp_path, law(searched))

    judged = advise(tmp_path)

    reused = {"verdict": "reuse", "requirement": None, "reason": where, "law": where}
    assert verdicts(judged, "law") == [reused | {"run": [where, where]}] * 2
    assert {v["verdict"] for v in verdicts(judged, "readings")} == {"reuse"}
    assert judged["candidates"]["a"]["store"] == str(tmp_path / "store")
    out = capsys.readouterr().out
    assert out.startswith(f"the reuse check at {'c0ffee' * 6}c0ff (origin/")
    assert f"  law server: reuse: {where}\n" in out


@pytest.mark.parametrize(
    "case, verdict, named",
    [
        ("another engine config", "calibrate", "no step linear law at its key"),
        ("a law file at another digest", "calibrate", "states ATOM digest 000000"),
        ("a store law at another digest", "calibrate", "never re-keyed"),
        ("no host laws beside a store law", "calibrate", "has no host laws beside"),
    ],
)
def test_a_law_at_another_key_names_requirement_1(
    tmp_path, searched, case, verdict, named
):
    r = searched
    if case == "another engine config":
        law_file(tmp_path, law(r, engine_config={**keyed(r.config), "ladder": [1]}))
    elif case == "a law file at another digest":
        held = law(r)
        held["host"]["key"]["atom"] = "0" * 64
        law_file(tmp_path, held)
    elif case == "a store law at another digest":
        stored(tmp_path, r, atom="0" * 64)
    else:
        stored(tmp_path, r, host=False)

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"]) for v in found] == [(verdict, 1)] * 2
    assert named in found[0]["reason"]


def test_a_law_file_s_host_laws_fitted_on_another_cpu_are_reused(tmp_path, searched):
    held = law(searched)
    held["host"]["key"]["host_cpu"] = "another CPU"
    where = law_file(tmp_path, held)

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["reason"]) for v in found] == [("reuse", where)] * 2


@pytest.mark.parametrize("calibrated", [False, True], ids=["reused", "calibrated"])
def test_a_law_file_s_readings_serve_only_where_its_law_is_reused(
    tmp_path, searched, calibrated
):
    # A calibration runs with the store law it publishes, which carries none.
    held = law(searched) | {memory.LAW_MEMBER: law_member(searched)}
    if calibrated:
        for source in held["sources"].values():
            source["figures"] = {}
    where = law_file(tmp_path, held)
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)

    judged = advise(tmp_path)

    laws = {(v["verdict"], v["requirement"]) for v in verdicts(judged, "law")}
    found = verdicts(judged, "readings")
    if not calibrated:
        assert laws == {("reuse", None)}
        reason = f"in {where}'s {memory.LAW_MEMBER}"
        assert [(v["verdict"], v["reason"]) for v in found] == [("reuse", reason)] * 2
        return
    assert laws == {("calibrate", 4)}
    assert [(v["verdict"], v["requirement"]) for v in found] == [("measure", 1)] * 2
    assert "no entry for memory_readings" in found[0]["reason"]


@pytest.mark.parametrize("invalidated", [True, False])
def test_a_store_law_the_store_refuses_to_load_names_requirement_2(
    tmp_path, searched, invalidated
):
    r = searched
    conditions = moved(readings(r), Axis.ATOM_SRC) if invalidated else None
    where = stored(tmp_path, r, conditions)

    found, _ = verdicts(advise(tmp_path), "law")

    if invalidated:
        assert (found["verdict"], found["requirement"]) == ("calibrate", 2)
        assert found["reason"].startswith(f"{where}: ")
        assert f"`{Kind.STEP_LINEAR_LAW}` x `{Axis.ATOM_SRC}`" in found["reason"]
    else:
        host = Key.of(Kind.HOST_LAWS, **host_laws_key(r)).token()
        assert found == {
            "verdict": "reuse",
            "requirement": None,
            "reason": where,
            "law": where,
            "run": [where.split()[1], host],
            "store": str(tmp_path / "store"),
        }


@pytest.mark.parametrize("bound, rungs", [(8, ["8"]), (12, ["8", "eager"])])
def test_a_scheduled_decode_rung_with_no_vector_names_requirement_3(
    tmp_path, searched, bound, rungs
):
    searched.config.max_num_seqs = bound
    publish_readings(tmp_path / "store", searched.config)
    held = law(searched)
    del held["decode"]["8"]
    law_file(tmp_path, held)

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"], v["rungs"]) for v in found] == [
        ("calibrate", 3, rungs)
    ] * 2
    listed = f"rung{'s' * (len(rungs) > 1)} {', '.join(rungs)}, which this"
    assert listed in found[0]["reason"]


@pytest.mark.parametrize(
    "records, verdict, named",
    [
        ("at the head", "refused", "on records run at the head"),
        ("unstated", "calibrate", "re-keyed stating no ATOM digest"),
        ("at another digest", "calibrate", "re-keyed from ATOM digest 0000"),
    ],
)
def test_a_figure_past_5_percent_names_requirement_4(
    tmp_path, searched, records, verdict, named
):
    held, run = past(searched)
    if records != "at the head":
        held["records_atom"] = "0" * 64 if records == "at another digest" else None
    law_file(tmp_path, held)

    found, _ = verdicts(advise(tmp_path), "law")

    assert (found["verdict"], found["requirement"]) == (verdict, 4)
    figure = f"run {run}, decode, median +5.12% of source refitted, past 5%"
    assert figure in found["reason"] and named in found["reason"]


#: A prefill-decode producer's transfer config, as the development template's
#: PP4 prefill side runs: it schedules no decode rung.
PRODUCER = {"kv_role": "kv_producer", "kv_connector": "compass"}


def staged(tmp_path, r, *medians, eager=True) -> dict:
    """`r`'s config on as many pipeline stages as `medians`, a producer as the
    template's prefill side, eager unless `eager` is false, with its readings
    in the store; and `law` re-keyed to it with a part per stage, as `fit
    steps` and `fit host` write one: each stage's decode cut, its figures on
    prefill within the gate but one run's median, `medians[k]` on stage k."""
    stages = len(medians)
    vars(r.config).update(
        pipeline_parallel_size=stages, enforce_eager=eager, kv_transfer_config=PRODUCER
    )
    publish_readings(tmp_path / "store", r.config)
    held = law(r)
    top = {k: held.pop(k) for k in ("platform", "engine_config", "records_atom")}
    host = held.pop("host")
    del held["sources"]
    parts = []
    for median in medians:
        figures = {run: {"prefill": dict(WITHIN)} for run in held["runs"]}
        figures[held["runs"][1]]["prefill"]["median"] = median
        parts.append(held | {"decode": {}, "figures": figures})
    laws = {k: v for k, v in host.items() if k != "key"}
    host = {"key": host["key"], "stages": [laws] * stages}
    return top | {"provenance": "planted", "stages": parts, "host": host}


@pytest.mark.parametrize(
    "medians, verdict",
    [
        ((-0.02, -0.02), ("reuse", None)),
        ((-0.02, 0.0512), ("refused", 4)),
        ((0.0512, -0.02), ("refused", 4)),
    ],
    ids=["both within", "stage 1 past", "stage 0 past"],
)
def test_a_pipeline_s_law_holds_each_stage_s_figures_to_5_percent_the_stage_named(
    tmp_path, searched, medians, verdict
):
    held = staged(tmp_path, searched, *medians)
    where = law_file(tmp_path, held)

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"]) for v in found] == [verdict] * 2
    if verdict[0] == "reuse":
        assert found[0]["reason"] == where
        return
    (stage,) = [k for k, median in enumerate(medians) if median > reuse.GATE]
    figure = f"run {held['stages'][stage]['runs'][1]}, prefill, median +5.12%"
    assert found[0]["reason"] == (
        f"{where}: {figure} of pipeline stage {stage}, past 5%, on records run at "
        "the head, which a calibration refits alike"
    )


@pytest.mark.parametrize("parts", ["laws", "stages"])
def test_a_pipeline_s_host_laws_under_another_key_are_refused_as_one_stage_s_are(
    tmp_path, searched, parts
):
    held = staged(tmp_path, searched, -0.02, -0.02)
    if parts == "laws":
        held["host"] = {"key": held["host"]["key"], **held["host"]["stages"][0]}
    held["host"]["key"]["engine"]["tensor_parallel_size"] = 2
    law_file(tmp_path, held)

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"]) for v in found] == [("refused", None)] * 2
    assert found[0]["reason"].startswith(
        "the run file's host laws were fitted under another "
        "['engine.tensor_parallel_size']; this run's key is "
    )


def test_a_law_with_a_source_stating_no_figures_names_requirement_4(tmp_path, searched):
    held = law(searched)
    del held["sources"]["lone"]["figures"]
    where = law_file(tmp_path, held)

    found, _ = verdicts(advise(tmp_path), "law")

    (made_by,) = found.pop("made_by")
    assert made_by.startswith("no calibrate.sh command: ")
    assert found == {
        "verdict": "calibrate",
        "requirement": 4,
        "reason": f"{where} states no cost-accuracy figures of source lone",
        "law": where,
        "run": [where, where],
    }


def host_alone(r) -> dict:
    """`law` with source shipped holding host laws alone and stating no
    figures, as the crs005 DPA8 prefill law's does, its prefill vector moved
    to source refitted."""
    held = law(r)
    shipped = held["sources"]["shipped"]
    shipped["vectors"].remove("prefill")
    del shipped["figures"]
    held["sources"]["refitted"]["vectors"].append("prefill")
    return held


def test_a_source_of_host_laws_alone_needs_no_figures(tmp_path, searched):
    where = law_file(tmp_path, host_alone(searched))

    found = verdicts(advise(tmp_path), "law")

    reused = {"verdict": "reuse", "requirement": None, "reason": where, "law": where}
    assert found == [reused | {"run": [where, where]}] * 2


def test_a_law_whose_sources_price_no_step_needs_its_own_figures(tmp_path, searched):
    held = host_alone(searched)
    held["sources"] = {"shipped": held["sources"]["shipped"]}
    where = law_file(tmp_path, held)

    found, _ = verdicts(advise(tmp_path), "law")

    (made_by,) = found.pop("made_by")
    assert made_by.startswith("no calibrate.sh command: ")
    assert found == {
        "verdict": "calibrate",
        "requirement": 4,
        "reason": f"{where} states no cost-accuracy figures",
        "law": where,
        "run": [where, where],
    }


@pytest.mark.parametrize("vectors", [["prefill"], ["prefill", "host"]])
def test_a_source_of_the_prefill_vector_needs_figures(tmp_path, searched, vectors):
    held = law(searched)
    shipped = held["sources"]["shipped"]
    shipped["vectors"] = vectors
    del shipped["figures"]
    where = law_file(tmp_path, held)

    found, _ = verdicts(advise(tmp_path), "law")

    (made_by,) = found.pop("made_by")
    assert made_by.startswith("no calibrate.sh command: ")
    assert found == {
        "verdict": "calibrate",
        "requirement": 4,
        "reason": f"{where} states no cost-accuracy figures of source shipped",
        "law": where,
        "run": [where, where],
    }


@pytest.mark.parametrize("held", ["none", "at another digest"])
def test_readings_with_no_entry_or_measured_at_another_digest_are_measured(
    tmp_path, searched, held
):
    law_file(tmp_path, law(searched))
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)
    if held == "at another digest":
        config = searched.config
        fields = memory.key_fields(config)
        platform = keyed_platform(Kind.MEMORY_READINGS, PLATFORM)
        ArtifactStore(tmp_path / "store").publish(
            Key.of(Kind.MEMORY_READINGS, **platform, **fields),
            provenance=STANZA,
            topology=memory.topology(config),
            conditions=moved(memory.conditions(fields, **PLATFORM), Axis.ATOM_SRC),
            gates=GateState.of(),
            members={
                member_name(memory.STEM, rank, memory.EXTENSION): json.dumps(
                    READINGS[1]
                ).encode()
                for rank in memory.topology(config).ranks()
            },
        )

    found = verdicts(advise(tmp_path), "readings")

    assert [(v["verdict"], v["requirement"]) for v in found] == [("measure", 1)] * 2
    if held == "none":
        assert "no entry for memory_readings" in found[0]["reason"]
    else:
        assert f"`{Kind.MEMORY_READINGS}` x `{Axis.ATOM_SRC}`" in found[0]["reason"]


@pytest.mark.parametrize(
    "index, reading, judged, state",
    [
        ("fp4", None, ("measure", 2), "missing readings"),
        ("fp4", 3, ("reuse", None), "searchable"),
        ("fp8", None, ("reuse", None), "searchable"),
    ],
    ids=["fp4-v41-without", "fp4-v41-with", "fp8-v41"],
)
def test_an_fp4_v41_entry_without_the_page_bound_reading_is_measured_as_check_finds_it(
    tmp_path, searched, index, reading, judged, state
):
    vars(searched.config).update(index_cache_dtype=index, hf_config=V41)
    member = READINGS[1] | ({} if reading is None else {PAGE_BOUND: reading})
    publish_readings(tmp_path / "store", searched.config, readings=member)
    held = law(searched)
    held["prefill"]["buckets"] = RECORDED
    law_file(tmp_path, held)

    found = verdicts(advise(tmp_path), "readings")

    assert {(v["verdict"], v["requirement"]) for v in found} == {judged}
    points = json.loads((tmp_path / "out" / STATES).read_text())["a"]["points"]
    assert {p["state"] for p in points.values()} == {state}
    if judged[0] == "measure":
        assert f"states no {PAGE_BOUND}, which an fp4 V4.1 start" in found[0]["reason"]
        (made,) = {c for p in points.values() for c in p["made_by"]}
        assert made.startswith("compass measure memory --gpu G --out D --store ")


def test_a_candidate_is_looked_up_in_the_latest_calibration_store_holding_all_it_reads(
    tmp_path, searched
):
    r = searched
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)
    # "a" holds every input; "b", published into later, the law alone.
    a, b = (tmp_path / f"out/cal/{name}/store" for name in "ab")
    for store in (a, b):
        platforms.flags(store, DOCUMENT, tmp_path)
        stored(tmp_path, r, store=store.relative_to(tmp_path))
        if store == a:
            publish_readings(a, r.config)

    judged = advise(tmp_path)

    assert judged["candidates"]["a"]["store"] == str(a)
    assert {v["verdict"] for v in verdicts(judged, "readings")} == {"reuse"}
    publish_readings(b, r.config)

    assert advise(tmp_path)["candidates"]["a"]["store"] == str(b)
    shutil.rmtree(tmp_path / "out/cal")

    judged = advise(tmp_path)

    assert judged["candidates"]["a"]["store"] == str(tmp_path / "store")
    assert {v["verdict"] for v in verdicts(judged, "readings")} == {"measure"}


@pytest.mark.parametrize(
    "case, looked_up, verdict",
    [
        ("every input at its key", "cal/a", ("refused", 4)),
        ("its law at another digest", "shared", ("calibrate", 1)),
        ("no platform readings", "shared", ("calibrate", 1)),
    ],
)
def test_a_calibration_store_holding_every_input_at_its_key_is_looked_up_whatever_its_verdict(
    tmp_path, searched, case, looked_up, verdict
):
    r = searched
    a = tmp_path / "out/cal/a/store"
    if case != "no platform readings":
        platforms.flags(a, DOCUMENT, tmp_path)
    atom = "0" * 64 if case == "its law at another digest" else None
    stored(tmp_path, r, atom=atom, store=a.relative_to(tmp_path), held=past(r)[0])
    publish_readings(a, r.config)

    judged = advise(tmp_path)

    store = a if looked_up == "cal/a" else tmp_path / "store"
    assert judged["candidates"]["a"]["store"] == str(store)
    found = verdicts(judged, "law")
    assert [(v["verdict"], v["requirement"]) for v in found] == [verdict] * 2
    assert {v["verdict"] for v in verdicts(judged, "readings")} == {"reuse"}


@pytest.mark.parametrize(
    "case, looked_up, verdict, named, readings_verdict",
    [
        ("a reusable law", "b", ("reuse", None), "", "measure"),
        ("host laws of another CPU", "b", ("reuse", None), "", "reuse"),
        ("past 5%", "b", ("refused", 4), "run at the head", "measure"),
        ("an older store holding every input", "a", ("reuse", None), "", "reuse"),
    ],
)
def test_with_no_store_holding_every_input_the_latest_a_law_is_read_from_is_looked_up(
    tmp_path, searched, case, looked_up, verdict, named, readings_verdict
):
    r = searched
    # "a" holds a reusable law, and in the last case its readings; "b", published
    # into later, the case's law, and in the case of another CPU its readings.
    a, b = (tmp_path / f"out/cal/{name}/store" for name in "ab")
    platforms.flags(a, DOCUMENT, tmp_path)
    stored(tmp_path, r, store=a.relative_to(tmp_path))
    if case == "an older store holding every input":
        publish_readings(a, r.config)
    platforms.flags(b, DOCUMENT, tmp_path)
    held = past(r)[0] if case == "past 5%" else law(r)
    if case == "host laws of another CPU":
        held["host"]["key"]["host_cpu"] = "another CPU"
        publish_readings(b, r.config)
    stored(tmp_path, r, store=b.relative_to(tmp_path), held=held)

    judged = advise(tmp_path)

    store = str(tmp_path / f"out/cal/{looked_up}/store")
    assert judged["candidates"]["a"]["store"] == store
    found = verdicts(judged, "law")
    assert [(v["verdict"], v["requirement"], v["store"]) for v in found] == [
        (*verdict, store)
    ] * 2
    assert named in found[0]["reason"]
    readings_found = {v["verdict"] for v in verdicts(judged, "readings")}
    assert readings_found == {readings_verdict}


def test_a_store_no_law_is_read_from_is_passed_over_for_the_shared_store(
    tmp_path, searched
):
    law_file(tmp_path, past(searched)[0])
    platforms.flags(tmp_path / "out/cal/x/store", DOCUMENT, tmp_path)

    judged = advise(tmp_path)

    assert judged["candidates"]["a"]["store"] == str(tmp_path / "store")
    found = verdicts(judged, "law")
    assert [(v["verdict"], v["requirement"]) for v in found] == [("refused", 4)] * 2
    assert {v["verdict"] for v in verdicts(judged, "readings")} == {"reuse"}


def test_each_1p1d_side_resolves_as_deploy_sh_starts_it(
    tmp_path, monkeypatch, at_head, v4_pro, resolved_once
):
    for name in PERFORMANCE_KEYS:
        monkeypatch.delenv(name, raising=False)
    cpu = read_json(PAIR["prefill"])["host"]["key"]["host_cpu"]
    monkeypatch.setattr(compass_run, "host_cpu", lambda: cpu)
    common = {"basis": "a recipe", "deployment": "1p1d", "env": SHARED_ENV}
    common["args"] = {"--tensor-parallel-size": 8, "--kv-cache-dtype": "fp8"}
    common["args"]["--enable-dp-attention"] = True
    common["prefill"] = {"args": {"--enable-tbo": True}, "env": SIDES["prefill"][1]}
    common["decode"] = {"args": {"--cudagraph-capture-sizes": DECODE_SIZES}}

    judged = advise(tmp_path, MACHINE, model=str(v4_pro), common=common)

    # The decode side lacks the rungs its ladder schedules (requirement 3); the
    # prefill side, a producer, schedules none, so its law is held to accuracy
    # (requirement 4), where a source of the pair's prefill law states none.
    for side, requirement in COVERAGE_REQUIREMENT.items():
        found = verdicts(judged, "law", side)
        assert [(v["verdict"], v["requirement"]) for v in found] == [
            ("calibrate", requirement)
        ] * 2
        uncovered = f"{PAIR[side]} has no decode vector for rungs {UNCOVERED}, which"
        assert (uncovered in found[0]["reason"]) == (side == "decode")
        (reason,) = {v["reason"] for v in verdicts(judged, "readings", side)}
        assert '"index_cache_dtype": "fp8"' in reason
        assert '"AITER_LOG_LEVEL": "WARNING"' in reason


@pytest.mark.parametrize(
    "exported, given, level",
    [
        ("", {}, "WARNING"),
        (None, {"AITER_LOG_LEVEL": ""}, "WARNING"),
        ("", {"AITER_LOG_LEVEL": "INFO"}, "INFO"),
    ],
    ids=["exported empty", "empty in the space", "set in the space"],
)
def test_a_1p1d_side_runs_under_the_aiter_log_level_deploy_sh_starts_it_with(
    tmp_path, monkeypatch, at_head, v4_pro, resolved_once, exported, given, level
):
    for name in PERFORMANCE_KEYS:
        monkeypatch.delenv(name, raising=False)
    if exported is not None:
        monkeypatch.setenv("AITER_LOG_LEVEL", exported)
    cpu = read_json(PAIR["prefill"])["host"]["key"]["host_cpu"]
    monkeypatch.setattr(compass_run, "host_cpu", lambda: cpu)
    common = {"basis": "a recipe", "deployment": "1p1d", "env": SHARED_ENV | given}
    common["args"] = {"--tensor-parallel-size": 8, "--kv-cache-dtype": "fp8"}
    common["args"]["--enable-dp-attention"] = True
    common["prefill"] = {"args": {"--enable-tbo": True}, "env": SIDES["prefill"][1]}
    common["decode"] = {"args": {"--cudagraph-capture-sizes": DECODE_SIZES}}

    judged = advise(tmp_path, MACHINE, model=str(v4_pro), common=common)

    # The pair's laws are keyed WARNING: each side finds its own there, else
    # none at its key.
    for side, own in COVERAGE_REQUIREMENT.items():
        requirement = own if level == "WARNING" else 1
        found = verdicts(judged, "law", side)
        assert [(v["verdict"], v["requirement"]) for v in found] == [
            ("calibrate", requirement)
        ] * 2
        (reason,) = {v["reason"] for v in verdicts(judged, "readings", side)}
        assert f'"AITER_LOG_LEVEL": "{level}"' in reason


def test_a_side_s_own_kv_transfer_config_is_refused_as_deploy_sh_refuses_it(
    tmp_path, searched
):
    law_file(tmp_path, law(searched))
    prefill = {"args": {"--kv-transfer-config": "{}"}}
    block = {"basis": "a recipe", "deployment": "1p1d", "prefill": prefill}

    judged = advise(tmp_path, common=block)

    named = "--kv-transfer-config is among the prefill side's ATOM arguments"
    for what in ("law", "readings"):
        found = verdicts(judged, what, "prefill")
        assert {(v["verdict"], v["requirement"]) for v in found} == {("refused", None)}
        assert found[0]["reason"].startswith(named)


def git(tree, *args) -> str:
    done = subprocess.run(
        ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def repository(tree, *versions) -> list[str]:
    """A repository at `tree` committing each of `versions`, an ``atom/``
    module's text, in turn, its integration branch's remote-tracking ref at the
    last; each commit."""
    git(tree.parent, "init", "-q", tree.name)
    git(tree, "remote", "add", "origin", str(tree))
    commits = []
    for text in versions:
        (tree / "atom").mkdir(exist_ok=True)
        (tree / "atom" / "x.py").write_text(text)
        git(tree, "add", "atom")
        git(tree, "commit", "-q", "-m", text)
        commits.append(git(tree, "rev-parse", "HEAD"))
    git(tree, "update-ref", f"refs/remotes/origin/{reuse.INTEGRATION}", commits[-1])
    return commits


@pytest.mark.parametrize(
    "case", ["no repository", "another digest", "no remote ref", "an unknown commit"]
)
def test_a_tree_that_is_not_the_head_refuses_every_verdict_by_name(
    tmp_path, monkeypatch, capsys, case
):
    tree = tmp_path / "tree"
    monkeypatch.setattr(reuse, "TREE", tree)
    tree.mkdir()
    commit = None
    if case != "no repository":
        (tip,) = repository(tree, "X = 1\n")
    if case == "no remote ref":
        git(tree, "update-ref", "-d", f"refs/remotes/origin/{reuse.INTEGRATION}")
    elif case == "an unknown commit":
        commit = "0123456"

    judged = advise(tmp_path, commit=commit) if commit else advise(tmp_path)

    reason = judged["refused"]["reason"]
    if case == "no repository":
        assert reason.startswith(f"the tree atom resolves under, {tree}, is not a git")
    elif case == "another digest":
        digest = compass_run.committed_atom_digest(tree, tip)
        assert f"has ATOM digest {compass_run.atom_digest()}, and" in reason
        assert f"the head, {tip} at origin/{reuse.INTEGRATION}, has {digest}" in reason
    elif case == "no remote ref":
        assert f"0 commits at a remote-tracking ref of {reuse.INTEGRATION}" in reason
    else:
        assert (
            f"the frozen commit 0123456 is not one of origin/{reuse.INTEGRATION}'s"
            in reason
        )
    assert judged["head"] is None and judged["candidates"]["a"]["store"] is None
    assert {
        (v["verdict"], v["requirement"], v["reason"])
        for what in ("law", "readings")
        for v in verdicts(judged, what)
    } == {("refused", 1, reason)}
    assert f"the reuse check refuses: {reason}\n" in capsys.readouterr().out


def test_the_head_is_the_frozen_commit_of_the_integration_branch_with_its_atom(
    tmp_path, monkeypatch
):
    tree = tmp_path / "tree"
    monkeypatch.setattr(reuse, "TREE", tree)
    tree.mkdir()
    frozen, tip = repository(tree, "X = 1\n", "X = 2\n")
    (tmp_path / "frozen" / "atom").mkdir(parents=True)
    (tmp_path / "frozen" / "atom" / "x.py").write_text("X = 1\n")
    git(tree, "checkout", "-q", "-b", "elsewhere", frozen)
    (tree / "atom" / "x.py").write_text("X = 3\n")
    git(tree, "commit", "-q", "-am", "off the branch")
    off = git(tree, "rev-parse", "HEAD")

    taken = reuse.head(frozen[:7])

    ref = f"origin/{reuse.INTEGRATION}"
    digest = compass_run.atom_digest(tmp_path / "frozen" / "atom")
    assert taken == {"commit": frozen, "ref": ref, "atom": digest}
    assert reuse.head()["commit"] == tip
    assert reuse.head()["atom"] != digest
    with pytest.raises(ValueError, match=f"the frozen commit {off} is not one of"):
        reuse.head(off)
