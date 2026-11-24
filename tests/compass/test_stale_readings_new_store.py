# SPDX-License-Identifier: MIT
"""A stale memory readings entry: the `compass measure memory` command that
`compass run --check` and `compass advice` print for it publishes into a new
store, as the store holding the entry refuses a second publish at its key, and
the reuse check looks the candidate up there once the readings have landed.

The entry is DeepSeek-V4.1's fp4 one without the PAGE bound, or one measured at
another ATOM digest; the run is `test_run_lacking`'s, the search
`test_advice_reuse`'s, its law a law file.
"""

import json
import re
import shutil

import platforms
import pytest
from test_advice_reuse import (  # noqa: F401
    advise,
    at_head,
    digested_once,
    law,
    law_file,
    searched,
    verdicts,
)
from test_drive_run import stub
from test_law_token_graph_buckets import RECORDED
from test_memory_readings import DOCUMENT, READINGS, publish_readings
from test_page_bound_reading import V41
from test_run_lacking import checked, lacked, served
from test_run_step_grid import deployment

from atom.compass import run as compass_run
from atom.compass.advice.check import STATES
from atom.compass.artifacts import ArtifactStore, Kind
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.drive import run as drive_run
from atom.compass.drive.lacking import MEASURED
from atom.compass.memory.readings import PAGE_BOUND

WHY = ["no PAGE bound", "another ATOM digest"]


def stale(store, config, why, monkeypatch):
    """`config`'s readings entry in `store`, left stale for `why`: an fp4
    V4.1 one stating no PAGE bound, or one an older tree measured."""
    if why == WHY[0]:
        vars(config).update(index_cache_dtype="fp4", hf_config=V41)
    with monkeypatch.context() as older:
        if why == WHY[1]:
            older.setattr(compass_run, "atom_digest", lambda root=None: "0" * 64)
            shutil.rmtree(store / Kind.MEMORY_READINGS.value)
        return publish_readings(store, config)


def new_store(beside) -> str:
    return f"{beside}@{compass_run.atom_digest()[:12]}/store"


def store_named(command: str) -> str:
    return re.search(r" --store (\S+) -- ", command)[1]


def holds(store, key) -> bool:
    return (ArtifactStore(store).directory_for(key) / ENTRY_FILE).is_file()


@pytest.mark.parametrize("why", WHY)
def test_check_measures_into_a_new_store_beside_its_out_which_then_serves(
    tmp_path, monkeypatch, capsys, why
):
    r = deployment(tmp_path)
    key = stale(tmp_path / "store", r.config, why, monkeypatch)
    served(monkeypatch, r.config)
    out = ["--step-stub", stub(tmp_path), "--out", str(tmp_path / "cal")]

    assert checked(tmp_path, *out) == 1

    said = capsys.readouterr().out
    assert lacked(said)[0].startswith("memory readings: stale")
    new = new_store(tmp_path / "cal")
    assert store_named(said) == new and not holds(new, key)
    publish_readings(new, r.config, readings=READINGS[1] | {PAGE_BOUND: 3})

    assert checked(tmp_path, *out) == 1

    assert f"made by: {MEASURED.format(new)}\n" in capsys.readouterr().out
    on = [*platforms.flags(new, DOCUMENT, tmp_path), "--store", new, "--check"]
    with pytest.raises(SystemExit) as exited:
        drive_run.main([*out, *on, "--", "--model", str(tmp_path)])
    assert exited.value.code == 0


@pytest.mark.parametrize("why", WHY)
def test_advice_measures_into_a_new_store_it_looks_the_candidate_up_in(
    tmp_path, searched, monkeypatch, why  # noqa: F811
):
    key = stale(tmp_path / "store", searched.config, why, monkeypatch)
    held = law(searched)
    if why == WHY[0]:
        held["prefill"]["buckets"] = RECORDED
    law_file(tmp_path, held)

    def made() -> set:
        states = json.loads((tmp_path / "out" / STATES).read_text())
        return {c for p in states["a"]["points"].values() for c in p["made_by"]}

    assert advise(tmp_path)["candidates"]["a"]["store"] == str(tmp_path / "store")

    (command,) = made()
    new = new_store(tmp_path / "out" / "cal" / "readings")
    assert store_named(command) == new and not holds(new, key)
    publish_readings(new, searched.config, readings=READINGS[1] | {PAGE_BOUND: 3})
    advise(tmp_path)

    assert made() == {MEASURED.format(new)}
    platforms.flags(new, DOCUMENT, tmp_path)

    judged = advise(tmp_path)

    assert judged["candidates"]["a"]["store"] == new
    assert {v["verdict"] for v in verdicts(judged, "readings")} == {"reuse"}
    assert made() == set()
