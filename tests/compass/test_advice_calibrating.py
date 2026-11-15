# SPDX-License-Identifier: MIT
"""`compass advice`'s check of a candidate missing a law: its `calibrate.sh`
command publishes into the shared store unless that store holds readings at
the configuration's key, at any ATOM digest, or a law of it at the head's,
each of which a second publish would meet; and a calibration into OUT's
``cal/<candidate>`` makes its candidate ``calibrating`` while its status file
ends in a step's start or end, and ``missing a law`` again, its command
resuming it, once that file ends in a failure or the calibration's end.

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
from test_artifact_invalidation import STANZA, moved
from test_memory_readings import READINGS
from test_run_step_grid import PLATFORM

from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    GateState,
    Key,
    Kind,
    keyed_platform,
    member_name,
)
from atom.compass.run import memory

#: A candidate with no readings at its key in the store, and one with them,
#: whose argument the stubbed resolution reads none of.
UNREAD, READ = {"args": {"--max-num-seqs": 12}}, {"args": {"--block-size": 16}}


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


@pytest.mark.parametrize("readings", ["none", "at another digest"])
def test_a_calibration_publishes_into_the_shared_store_unless_it_holds_its_readings(
    tmp_path, stubbed, readings  # noqa: F811
):
    store = tmp_path / "store"
    if readings == "at another digest":
        config = resolved(stubbed.config, ["--max-num-seqs", "12"])
        _readings_at_another_digest(store, config)

    found = advise(tmp_path, {"a": READ, "b": UNREAD})

    assert found["a"]["state"] == found["b"]["state"] == "missing a law"
    assert "STORE" not in _env(found, "a")
    published = _env(found, "b").get("STORE")
    assert published == (str(store) if readings == "none" else None)
    reused = json.loads((tmp_path / "out" / "reuse.json").read_text())
    held = reused["candidates"]["b"]["points"]["b-c1"]["readings"]["server"]
    assert (held["verdict"], held["held"]) == ("measure", readings != "none")


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
    else:
        (command,) = found["b"]["made_by"]
        assert f"calibrate.sh {status.parent} " in command
    assert starts(tmp_path) == []
    printed = capsys.readouterr().out
    assert f"  state: {state}\n    calibration: {said}\n" in printed
