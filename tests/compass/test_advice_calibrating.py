# SPDX-License-Identifier: MIT
"""`compass advice`'s check of a candidate missing a law: its `calibrate.sh`
command publishes into the shared store unless that store holds readings at
the configuration's key, at any ATOM digest, or a law of it at the head's,
each of which a second publish would meet, until a calibration has started:
its command then keeps the store the calibration's inputs name, so a resume
takes the same inputs; and a calibration into OUT's ``cal/<candidate>`` makes
its candidate ``calibrating``, its command the one that resumes it once none
runs, while its status file ends in a step's start or end, and ``missing a
law`` again, its command resuming it, once that file ends in a failure or the
calibration's end.

Two 1P1D candidates sharing a prefill side, as the development template's A
and B, take one command each: A's fits both sides, and B's its decode alone
into the store A's publishes into, naming A's calibration for its prefill side;
once that store holds A's two sides and B's decode, both are searchable on it.
A started calibration's command holds once one of its sides has published, and
B, once A has ended in a store of its own, fits its decode alone there.

The configuration, its stubbed resolution and starts are `test_advice_check`'s:
the store holds readings of the configuration as the deployment resolves it,
and no law of it.
"""

import json
import shlex
import shutil
from types import SimpleNamespace

import platforms
import pytest
from test_advice_check import (
    NETWORK,
    advise,
    resolved,
    starts,
    stubbed,  # noqa: F401  the fixture
)
from test_advice_reuse import stored
from test_artifact_invalidation import STANZA, moved
from test_memory_readings import READINGS, publish_readings
from test_run_step_grid import DOCUMENT, PLATFORM

from atom.compass.advice import check
from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    GateState,
    Key,
    Kind,
    keyed_platform,
    member_name,
)
from atom.compass.backends.step_linear_plan import FIRST_TURN, TRACE_RUNS
from atom.compass.drive import lacking, reuse
from atom.compass.run import memory

#: A candidate with no readings at its key in the store, and one with them,
#: whose argument the stubbed resolution reads none of; the first's
#: --max-num-seqs caps its capture ladder below the other's, so their laws'
#: keys differ.
UNREAD, READ = {"args": {"--max-num-seqs": 4}}, {"args": {"--block-size": 16}}


def _env(found: dict, name: str) -> dict:
    """The variables of `name`'s one printed calibrate.sh command."""
    (command,) = found[name]["made_by"]
    words = shlex.split(command.split("\n")[0])
    return dict(w.split("=", 1) for w in words if "=" in w and w[0].isupper())


def _readings_at_another_digest(store, config) -> None:
    """`config`'s readings at their key in `store`, measured at another ATOM
    source than this tree's."""
    fields = memory.key_fields(config)
    ArtifactStore(store).publish(
        Key.of(
            Kind.MEMORY_READINGS,
            **keyed_platform(Kind.MEMORY_READINGS, PLATFORM),
            **fields,
        ),
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


@pytest.mark.parametrize(
    "readings", ["none", "at another digest", "without the PAGE bound"]
)
def test_a_calibration_publishes_into_the_shared_store_unless_it_holds_its_readings(
    tmp_path, stubbed, readings, monkeypatch  # noqa: F811
):
    store = tmp_path / "store"
    config = resolved(stubbed.config, ["--max-num-seqs", "4"])
    if readings == "at another digest":
        _readings_at_another_digest(store, config)
    if readings == "without the PAGE bound":
        # Readings at the head's digest that a start of the config finds
        # stale, as an fp4 DeepSeek-V4.1 start finds readings with no PAGE bound.
        publish_readings(store, config)
        monkeypatch.setattr(reuse, "stale_page_bound", lambda read, _: "no PAGE")

    found = advise(tmp_path, {"a": READ, "b": UNREAD})

    assert found["a"]["state"] == found["b"]["state"] == "missing a law"
    assert "STORE" not in _env(found, "a")
    published = _env(found, "b").get("STORE")
    assert published == (str(store) if readings == "none" else None)
    reused = json.loads((tmp_path / "out" / "reuse.json").read_text())
    held = reused["candidates"]["b"]["points"]["b-c1"]["readings"]["server"]
    assert (held["verdict"], held["held"]) == ("measure", readings != "none")


def test_a_calibration_command_runs_under_the_candidate_s_own_variables(
    tmp_path, stubbed  # noqa: F811
):
    # A variable no law key states, as MiniMax-M3's ATOM_M3_INDEXER_CP, which
    # moves the engine config the law is keyed by.
    found = advise(tmp_path, {"a": READ, "b": UNREAD | {"env": {"STUB_OWN": "1"}}})

    assert _env(found, "b")["STUB_OWN"] == "1"
    assert "STUB_OWN" not in _env(found, "a")


@pytest.mark.parametrize("started", ["into the shared store", "into its own"])
def test_a_started_calibration_s_command_keeps_the_store_its_inputs_name(
    tmp_path, stubbed, started  # noqa: F811
):
    store = tmp_path / "store"
    before = _env(advise(tmp_path, {"a": READ, "b": UNREAD}), "b")
    # calibrate.sh's inputs, then its memory step's readings in the store it
    # publishes into, then a failed record.
    cal = tmp_path / "out" / "cal" / "b"
    cal.mkdir(parents=True, exist_ok=True)
    given = ["tree c0ffee", "record 1"]
    if started == "into the shared store":
        given.append(f"store {store.resolve()}")
        config = resolved(stubbed.config, ["--max-num-seqs", "4"])
        publish_readings(store, config)
    (cal / "inputs").write_text("".join(f"{line}\n" for line in given))
    taken = ["start memory", "end memory 60 s", "start cal1-c1", "fail cal1-c1 9 s: x"]
    (cal / "status").write_text("".join(f"2026-11-16T07:00:00Z {t}\n" for t in taken))

    found = advise(tmp_path, {"a": READ, "b": UNREAD})

    assert before["STORE"] == str(store)
    assert found["b"]["state"] == "missing a law"
    after = _env(found, "b")
    if started == "into the shared store":
        reused = json.loads((tmp_path / "out" / "reuse.json").read_text())
        held = reused["candidates"]["b"]["points"]["b-c1"]["readings"]["server"]
        assert held["verdict"] == "reuse"
        assert after == before | {"STORE": str(store.resolve())}
    else:
        assert after == {k: v for k, v in before.items() if k != "STORE"}


@pytest.mark.parametrize(
    "last, state, ended",
    [
        ("start cal2-c4", "calibrating", 1),
        ("end cal2-c4 12 s", "calibrating", 2),
        ("fail cal2-c4 12 s: client.sh exit 1; see x", "missing a law", 1),
        ("end calibration 0.17 node-hours", "missing a law", 1),
    ],
)
def test_a_calibration_s_status_makes_it_calibrating_until_it_fails_or_ends(
    tmp_path, stubbed, capsys, last, state, ended  # noqa: F811
):
    status = tmp_path / "out" / "cal" / "b" / "status"
    status.parent.mkdir(parents=True)
    taken = ["start warm", "end warm 600 s", last]
    status.write_text("".join(f"2026-11-15T07:00:00Z {line}\n" for line in taken))

    found = advise(tmp_path, {"a": READ, "b": UNREAD})

    said = f"{status}, {ended} steps ended, last: 2026-11-15T07:00:00Z {last}"
    assert found["a"]["state"] == "missing a law"
    assert "calibration" not in found["a"]
    assert found["b"]["state"] == state
    assert found["b"]["calibration"] == said
    assert found["b"]["why"][-1] == said
    if state == "calibrating":
        assert found["b"]["made_by"] == []
        (command,) = found["b"]["resumed_by"]
        shown = f"    resumed, once no calibrate.sh runs, by: {command}\n"
    else:
        assert "resumed_by" not in found["b"]
        (command,) = found["b"]["made_by"]
        shown = f"    made by: {command}\n"
    assert f"calibrate.sh {status.parent} " in command
    assert starts(tmp_path) == []
    printed = capsys.readouterr().out
    assert f"  state: {state}\n    calibration: {said}\n{shown}" in printed


#: Two 1P1D candidates sharing a prefill side, their decode sides keyed apart
#: by a performance variable.
PREFILL = {"args": {"--max-num-seqs": 8}, "env": {"STUB_PREFILL": "1"}}
PAIR = {
    name: {
        "deployment": "1p1d",
        "prefill": PREFILL,
        "decode": {"args": {"--max-num-seqs": 8}, "env": {"GPU_MAX_HW_QUEUES": q}},
    }
    for name, q in (("A", "2"), ("B", "4"))
}


@pytest.fixture
def sides(tmp_path, monkeypatch, stubbed):  # noqa: F811
    """The store holding no readings, every side under ``AITER_LOG_LEVEL``
    WARNING, as deploy.sh starts it, and none under this process's
    ``GPU_MAX_HW_QUEUES``; the config each side resolves to."""
    monkeypatch.setenv("AITER_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("GPU_MAX_HW_QUEUES", raising=False)
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)
    return resolved(stubbed.config, ["--max-num-seqs", "8"])


def _calibrate(found: dict, name: str) -> tuple[dict, list]:
    """The variables and the arguments of `name`'s one printed calibrate.sh
    command."""
    (command,) = [c for c in found[name]["made_by"] if "calibrate.sh " in c]
    words = shlex.split(command.split("\n")[0])
    at = words.index(str(lacking.CALIBRATE))
    return dict(w.split("=", 1) for w in words[:at]), words[at + 1 :]


@pytest.mark.parametrize(
    "held, started",
    [(False, False), (True, False), (True, True)],
    ids=["into the shared store", "into A's own, not started", "into A's own"],
)
def test_two_1p1d_candidates_sharing_a_prefill_side_fit_it_once(
    tmp_path, sides, held, started
):
    store, cal = tmp_path / "store", tmp_path / "out" / "cal"
    if held:
        # The shared store refuses A's prefill readings: A fits into its own.
        _readings_at_another_digest(store, sides)
    if started:
        (cal / "A").mkdir(parents=True)
        (cal / "A" / "inputs").write_text("tree c0ffee\nrecord 1\n")

    found = advise(tmp_path, PAIR, network=NETWORK)

    assert found["A"]["state"] == found["B"]["state"] == "missing a law"
    into = str(cal / "A" / "store" if held else store)
    env, words = _calibrate(found, "A")
    rungs = [f"{n} {FIRST_TURN}" for n in (1, 2, 4, 8)]
    assert words == [str(cal / "A"), *rungs, *TRACE_RUNS]
    assert env.get("STORE") == (None if held else into)
    assert {k: env.get(k) for k in ("FIT", "PREFILL_GPU", "DECODE_GPU")} == {
        "FIT": "prefill decode",
        "PREFILL_GPU": "0",
        "DECODE_GPU": "1",
    }
    assert env["PREFILL_ARGS"] == env["DECODE_ARGS"] == "--max-num-seqs 8"
    assert (env["AITER_LOG_LEVEL"], env["PREFILL_ENV"]) == ("WARNING", "STUB_PREFILL=1")
    assert env["DECODE_ENV"] == "GPU_MAX_HW_QUEUES=2"
    assert env["ROUTER_ARGS"] == " ".join(check.ROUTER)
    # B names A's calibration for its prefill side, whose readings A measures,
    # and fits its decode alone into that store once it may publish there.
    borrowed = check.BORROWED.format("A", into)
    if held and not started:
        assert found["B"]["made_by"] == [borrowed, check.WAITING.format(into, "A")]
        return
    assert found["B"]["made_by"][0] == borrowed and len(found["B"]["made_by"]) == 2
    env, words = _calibrate(found, "B")
    assert words == [str(cal / "B"), *rungs]
    assert (env["FIT"], env["STORE"], env["DECODE_ENV"]) == (
        "decode",
        into,
        "GPU_MAX_HW_QUEUES=4",
    )


def test_both_1p1d_candidates_are_searchable_once_one_store_holds_their_sides(
    tmp_path, sides
):
    # A's two sides and B's decode side in the shared store, each keyed under
    # its side's variables; B's prefill side is A's.
    for env in ({}, {"GPU_MAX_HW_QUEUES": "2"}, {"GPU_MAX_HW_QUEUES": "4"}):
        with check._env(env):
            stored(tmp_path, SimpleNamespace(config=sides))
            publish_readings(tmp_path / "store", sides)

    found = advise(tmp_path, PAIR, network=NETWORK)

    assert found["A"]["state"] == found["B"]["state"] == "searchable"
    reused = json.loads((tmp_path / "out" / "reuse.json").read_text())
    held = {name: c["store"] for name, c in reused["candidates"].items()}
    assert held == {"A": str(tmp_path / "store"), "B": str(tmp_path / "store")}
    assert len(starts(tmp_path)) == 2


def test_a_started_1p1d_calibration_s_command_holds_once_a_side_has_published(
    tmp_path, sides
):
    store, a = tmp_path / "store", tmp_path / "out" / "cal" / "A"
    first = advise(tmp_path, PAIR, network=NETWORK)
    # A started into the shared store, which then holds both sides' readings
    # and the prefill side's laws: its decode side's fit failed.
    a.mkdir(parents=True)
    (a / "inputs").write_text(f"tree c0ffee\nfit prefill decode\nstore {store}\n")
    for env in ({}, {"GPU_MAX_HW_QUEUES": "2"}):
        with check._env(env):
            publish_readings(store, sides)
            if not env:
                stored(tmp_path, SimpleNamespace(config=sides))

    found = advise(tmp_path, PAIR, network=NETWORK)

    assert found["A"]["state"] == found["B"]["state"] == "missing a law"
    # A resumes on the inputs it started with, the trace runs included; B
    # fits its decode alone into that store, its prefill side now reused.
    assert _calibrate(found, "A") == _calibrate(first, "A")
    assert _calibrate(found, "B") == _calibrate(first, "B")
    assert len(found["B"]["made_by"]) == 1


def test_a_1p1d_side_an_ended_calibration_published_leaves_the_other_to_fit_there(
    tmp_path, sides
):
    store, cal = tmp_path / "store", tmp_path / "out" / "cal"
    own = cal / "A" / "store"
    # The shared store refuses A's prefill readings, so A has ended in its own
    # store, which holds its two sides' laws and readings and the platform's.
    _readings_at_another_digest(store, sides)
    own.mkdir(parents=True)
    (cal / "A" / "inputs").write_text("tree c0ffee\nfit prefill decode\n")
    (cal / "A" / "status").write_text("2026-11-24T00:00:00Z end calibration 0.50\n")
    platforms.flags(own, DOCUMENT, tmp_path)
    for env in ({}, {"GPU_MAX_HW_QUEUES": "2"}):
        with check._env(env):
            stored(tmp_path, SimpleNamespace(config=sides), store=own)
            publish_readings(own, sides)

    found = advise(tmp_path, PAIR, network=NETWORK)

    assert (found["A"]["state"], found["B"]["state"]) == ("searchable", "missing a law")
    # B reuses A's prefill side in A's store, and fits its decode alone there.
    assert len(found["B"]["made_by"]) == 1
    env, words = _calibrate(found, "B")
    assert words == [str(cal / "B"), *[f"{n} {FIRST_TURN}" for n in (1, 2, 4, 8)]]
    assert (env["FIT"], env["STORE"]) == ("decode", str(own))
