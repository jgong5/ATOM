# SPDX-License-Identifier: MIT
"""`compass advice`'s check of a candidate missing a law: its `calibrate.sh`
command publishes into the shared store unless that store holds readings at
the configuration's key, at any ATOM digest, or a law of it at the head's,
each of which a second publish would meet, until a calibration has started:
its command then keeps the store the calibration's inputs name, so a resume
takes the same inputs; and a calibration into OUT's ``cal/<candidate>`` makes
its candidate ``calibrating``, its command the one that resumes it once none
runs, while its status file ends in a step's start or end, and ``missing a
law`` again, its command resuming it, once that file ends in a failure. Once
it ends in the calibration's end, the next calibration goes into
``cal/<candidate>+2``, unless it would take what the ended one took and refit
its law, at the candidate's key at the head's ATOM digest.

The configuration, its stubbed resolution and starts are `test_advice_check`'s:
the store holds readings of the configuration as the deployment resolves it,
and no law of it.
"""

import json
import shlex

import pytest
from test_advice_check import (
    advise,
    resolved,
    starts,
    stubbed,  # noqa: F401  the fixture
)
from test_advice_results import _calibrated
from test_artifact_invalidation import STANZA, moved
from test_memory_readings import READINGS, publish_readings
from test_run_step_grid import PLATFORM

from atom.compass import run as compass_run
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
from atom.compass.drive import lacking, reuse
from atom.compass.run import memory
from atom.compass.run.step_grid import keyed

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


def _fitted(directory, config, atom=None, platform=PLATFORM) -> None:
    """The law the stub's fit wrote in `directory`, keyed as `compass fit steps`
    keys it: `config`'s model and engine config, under this process's
    environment, its records at the ATOM digest `atom`, else this tree's, on
    `platform`."""
    law = directory / check.FITTED
    held = json.loads(law.read_text())
    held["key"] = {
        "model": compass_run._file_digest(config.model, "config.json"),
        "engine_config": json.dumps(keyed(config), sort_keys=True),
        "source_root": compass_run.source_root(atom),
        "form": "history",
        **keyed_platform(Kind.STEP_LINEAR_LAW, platform),
    }
    law.write_text(json.dumps(held))


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
    ],
)
def test_a_calibration_s_status_makes_it_calibrating_until_it_fails(
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


@pytest.mark.parametrize(
    "since",
    [
        "the tree moved",
        "its arguments changed",
        "its environment changed",
        "it ran on another device",
        "nothing",
    ],
)
def test_an_ended_calibration_s_candidate_calibrates_anew_unless_that_refits_its_law(
    tmp_path, stubbed, monkeypatch, since  # noqa: F811
):
    """b's printed calibration, run by `calibrate.sh` in `test_calibrate`'s
    stub tree at its commit ``cafe``, ends, its law keyed as a fit keys it. The
    next search prints the next calibration into ``cal/b+2``, where it records,
    its status read for b's state, or why none serves where it would take what
    the ended one took and that law is at b's key at the head's ATOM digest."""
    monkeypatch.setenv("AIPERF_PYTHON", "/real/python")
    cal, b = tmp_path / "out" / "cal", UNREAD
    (first,) = advise(tmp_path, {"b": b})["b"]["made_by"]
    assert _calibrated(tmp_path, first)[0] == 0
    atom = "0" * 64 if since == "the tree moved" else None
    # The law a calibration of b on another card fits, OUT then taken here.
    elsewhere = {"device": "MI300X gfx942"} if since.endswith("device") else {}
    config = resolved(stubbed.config, ["--max-num-seqs", "4"])
    _fitted(cal / "b", config, atom, PLATFORM | elsewhere)
    if since == "the tree moved":
        (tmp_path / "ATOM" / ".compass-commit").write_text("beef\n")
    elif since == "its arguments changed":
        # One the stubbed resolution reads none of: its inputs move, its key not.
        b = {"args": UNREAD["args"] | READ["args"]}
    elif since == "its environment changed":
        b = UNREAD | {"env": {"ATOM_USE_TRITON_MOE": "1"}}

    found = advise(tmp_path, {"b": b})["b"]

    assert (found["state"], "calibration" in found) == ("missing a law", False)
    (command,) = found["made_by"]
    if since == "nothing":
        ended = cal / "b"
        assert command == check.REFITS.format(ended, check.started_store(ended))
        return
    assert f" {lacking.CALIBRATE} {cal / 'b+2'} " in command
    # Its first record fails, and the next search reads cal/b+2's status.
    assert _calibrated(tmp_path, command, CLIENT_FAIL="cal1-c")[0] == 1
    stopped = advise(tmp_path, {"b": b})["b"]
    assert stopped["calibration"].startswith(f"{cal / 'b+2' / 'status'}, ")
    (resumed,) = stopped["made_by"]
    status, log = _calibrated(tmp_path, resumed)
    assert status == 0
    served = [line.split(" --out ")[1] for line in log if line.startswith("record ")]
    assert served[0].startswith(f"{cal / 'b+2' / 'cal1-c'}")
    assert all(s.startswith(f"{cal / 'b+2'}/") for s in served)
