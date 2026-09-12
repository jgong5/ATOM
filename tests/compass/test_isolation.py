"""The isolation audit, against the file the run actually writes.

`gpu_watch.sh` emits one JSON object per sample, `{"t", "visible", "phase",
"own_pids", "pids", "smi"}`, where `smi` is a `rocm-smi --json` blob. These
build that shape by hand rather than asserting on the auditor's internals, so a
change to what the sampler writes breaks the test that consumes it.

The two verdicts are tested apart, because they mean different things: a card
this run owned being somebody else's invalidates its memory readings (exit 1),
while a busy card it did not own cannot enter those readings at all and only
costs it the host (exit 2).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


isolation = _load("isolation")


def _sample(t: str, visible: str, cards: dict, phase: str = "unknown",
            own_pids=(), pids=(), bytes_used=None) -> str:
    """One sampler line.

    `cards` maps index -> (use %, VRAM %). `bytes_used` optionally maps index
    -> absolute bytes, which is what `--showmeminfo vram` adds and what the
    rounded percent cannot express.
    """
    smi = {f"card{i}": {"Device ID": "0x74a2", "GUID": str(1000 + i),
                        "GPU use (%)": str(use),
                        "GPU Memory Allocated (VRAM%)": str(vram)}
           for i, (use, vram) in cards.items()}
    for i, used in (bytes_used or {}).items():
        smi[f"card{i}"]["VRAM Total Used Memory (B)"] = str(used)
    return json.dumps({"t": t, "visible": visible, "phase": phase,
                       "own_pids": [str(p) for p in own_pids],
                       "pids": {"system": {f"PID{p}": "unknown, 0, 0, 0, 0"
                                           for p in pids}},
                       "smi": smi})


def _write(tmp_path, samples) -> str:
    path = tmp_path / "gpu.jsonl"
    path.write_text("\n".join(samples) + "\n")
    return str(path)


def _quiet(n: int) -> dict:
    return {i: (0, 0) for i in range(n)}


class TestAQuietBoxPasses:
    def test_own_cards_working_alone_is_clean(self, tmp_path, capsys):
        """The baseline is empty cards; everything after it is our engine."""
        working = {**_quiet(8), 0: (100, 70), 1: (98, 70)}
        path = _write(tmp_path,
                      [_sample("t0", "0,1", _quiet(8), phase="baseline")]
                      + [_sample(f"t{i}", "0,1", working, phase="real-server")
                         for i in range(1, 4)])
        assert isolation.main([path]) == 0
        out = capsys.readouterr().out
        assert "CLEAN" in out and "up to 100% use" in out

    def test_our_own_pids_are_not_other_tenants(self, tmp_path):
        path = _write(tmp_path, [
            _sample("t0", "0", _quiet(4), phase="baseline"),
            _sample("t1", "0", {**_quiet(4), 0: (99, 70)}, phase="real-server",
                    own_pids=(4242,), pids=(4242,))])
        assert isolation.audit(isolation.read(path))["verdict"] == "clean"

    def test_the_audit_is_written_where_asked(self, tmp_path):
        path = _write(tmp_path, [_sample("t0", "0", _quiet(4),
                                         phase="baseline")])
        out = tmp_path / "audit.json"
        isolation.main([path, "--json", str(out)])
        report = json.loads(out.read_text())
        assert report["owned"] == [0] and report["verdict"] == "clean"
        assert report["baseline_provenance"] == "phase-stamped"


class TestANeighbourCostsTheHostNotTheBudget:
    def test_a_busy_card_the_run_does_not_own_is_a_node_finding(
            self, tmp_path, capsys):
        """Card 4's bytes are not in card 0's `mem_get_info`."""
        cards = {**_quiet(8), 4: (100, 58), 5: (100, 58)}
        path = _write(tmp_path, [
            _sample("t0", "0,1", cards, phase="baseline"),
            _sample("t1", "0,1", cards, phase="real-server"),
            _sample("t2", "0,1", cards, phase="real-server")])
        assert isolation.main([path]) == 2
        out = capsys.readouterr().out
        assert "NODE NOT QUIET" in out and "4 (3/3 samples" in out
        assert "memory readings stand" in out

    def test_a_memory_run_may_waive_the_busy_node(self, tmp_path):
        cards = {**_quiet(8), 7: (100, 58)}
        path = _write(tmp_path, [_sample("t0", "0", cards, phase="baseline")])
        assert isolation.main([path, "--allow-busy-node"]) == 0

    def test_a_neighbour_that_arrives_midway_is_still_caught(self, tmp_path):
        """What a before/after pair misses."""
        quiet, busy = _quiet(8), {**_quiet(8), 7: (100, 60)}
        path = _write(tmp_path, [_sample("t0", "0", quiet, phase="baseline"),
                                 _sample("t1", "0", busy, phase="real-server"),
                                 _sample("t2", "0", quiet, phase="real-server")])
        assert isolation.main([path]) == 2
        report = isolation.audit(isolation.read(path))
        assert report["busy_neighbours"]["7"]["samples"] == 1
        assert report["own_clean"] is True

    def test_one_activity_sample_with_no_visible_memory_is_unknown(
            self, tmp_path, capsys):
        """The reading from the first clean matrix run, left unresolved.

        Cards 4-7 read 5-100% use at 0% VRAM for one sample of 31. That is not
        proof of a neighbour and it is not proof of quiet either: on a 192 GiB
        card `rocm-smi` rounds a 0.7 GiB compute-bound process to 0%.
        """
        quiet = _quiet(8)
        flicker = {**quiet, 4: (100, 0), 5: (6, 0)}
        path = _write(tmp_path, [_sample("t0", "0,1,2,3", quiet,
                                         phase="baseline"),
                                 _sample("t1", "0,1,2,3", flicker,
                                         phase="real-server"),
                                 _sample("t2", "0,1,2,3", quiet,
                                         phase="real-server")])
        assert isolation.main([path]) == 3
        out = capsys.readouterr().out
        assert "UNKNOWN" in out and "NOT CLASSIFIED" in out

    def test_sustained_activity_with_no_visible_memory_is_a_neighbour(
            self, tmp_path):
        """Rounded to 0%, but there twice running: somebody is working."""
        quiet = _quiet(8)
        busy = {**quiet, 4: (100, 0)}
        path = _write(tmp_path, [_sample("t0", "0", quiet, phase="baseline"),
                                 _sample("t1", "0", busy, phase="real-server"),
                                 _sample("t2", "0", busy, phase="real-server")])
        assert isolation.main([path]) == 2

    def test_the_drivers_own_reservation_is_not_a_tenant(self, tmp_path):
        """Every card on an idle MI308X node holds ~284 MiB and always has.

        Sampled with nothing of ours started, seven cards read 297,779,200 B
        and the eighth 297,783,296 B -- identical across cards nobody was
        using. Charged as occupancy it makes every run on the node
        contaminated, including its own baseline, which is how the first-use
        gate probe came back rc=1 on a card it had to itself.
        """
        floor = {i: int(297_779_200) for i in range(8)}
        floor[0] = 297_783_296
        path = _write(tmp_path, [
            _sample("t0", "0", _quiet(8), phase="baseline", bytes_used=floor),
            _sample("t1", "0", {**_quiet(8), 0: (100, 70)},
                    phase="real-server",
                    bytes_used={**floor, 0: int(150 * (1 << 30))})])
        report = isolation.audit(isolation.read(path))
        assert report["verdict"] == "clean" and report["own_clean"] is True

    def test_absolute_bytes_beat_the_rounded_percent(self, tmp_path):
        """0.7 GiB on a 192 GiB card reports 0% VRAM and is still a tenant."""
        quiet = _quiet(8)
        small = {**quiet, 4: (100, 0)}
        path = _write(tmp_path, [
            _sample("t0", "0", quiet, phase="baseline"),
            _sample("t1", "0", small, phase="real-server",
                    bytes_used={4: int(0.7 * (1 << 30))})])
        report = isolation.audit(isolation.read(path))
        assert report["verdict"] == "node_busy"
        assert "0.7 GiB" in report["problems"][0]

    def test_a_process_that_is_not_ours_is_a_node_finding(self, tmp_path):
        path = _write(tmp_path, [
            _sample("t0", "0", _quiet(4), phase="baseline", pids=(99,)),
            _sample("t1", "0", _quiet(4), phase="real-server",
                    own_pids=(4242,), pids=(99, 4242))])
        report = isolation.audit(isolation.read(path))
        assert report["verdict"] == "node_busy"
        assert report["foreign_pids"] == ["99"]


class TestOurOwnDevicesBeingSomebodyElses:
    def test_an_own_card_active_at_the_baseline_with_no_memory_is_unknown(
            self, tmp_path, capsys):
        """Nothing of ours exists yet, but the percent cannot rule it out."""
        path = _write(tmp_path, [
            _sample("t0", "0", {**_quiet(4), 0: (40, 0)}, phase="baseline"),
            _sample("t1", "0", {**_quiet(4), 0: (99, 70)}, phase="real-server")])
        assert isolation.main([path]) == 3
        assert "no memory visible" in capsys.readouterr().out

    def test_an_own_card_occupied_at_the_baseline_fails(self, tmp_path, capsys):
        """The engine holds nothing at the baseline, so this is not ours."""
        busy = {**_quiet(4), 0: (0, 58)}
        path = _write(tmp_path, [
            _sample("t0", "0", busy, phase="baseline"),
            _sample("t1", "0", {**busy, 0: (100, 70)}, phase="real-server")])
        assert isolation.main([path]) == 1
        out = capsys.readouterr().out
        assert "SELECTED DEVICES CONTAMINATED" in out
        assert "already in use at the baseline" in out

    def test_our_own_startup_is_not_another_tenant(self, tmp_path):
        """A card filling up after the baseline is this run filling it."""
        path = _write(tmp_path, [
            _sample("t0", "0", _quiet(4), phase="baseline"),
            _sample("t1", "0", {**_quiet(4), 0: (0, 40)}, phase="real-server"),
            _sample("t2", "0", {**_quiet(4), 0: (99, 70)}, phase="real-server")])
        report = isolation.audit(isolation.read(path))
        assert report["verdict"] == "clean" and report["own_clean"] is True

    def test_without_a_phase_stamp_the_first_sample_is_used_and_labelled(
            self, tmp_path):
        """Older audits have no phases; the weaker baseline is named as such."""
        path = tmp_path / "gpu.jsonl"
        path.write_text(json.dumps({
            "t": "t0", "visible": "0",
            "smi": {"card0": {"GPU use (%)": "0",
                              "GPU Memory Allocated (VRAM%)": "58"}}}) + "\n")
        report = isolation.audit(isolation.read(str(path)))
        assert report["baseline_provenance"] == "first-sample"
        assert report["verdict"] == "own_contaminated"


class TestTheFileItselfIsHandled:
    def test_a_truncated_final_line_does_not_lose_the_run(self, tmp_path):
        """The sampler is killed mid-write when the cell ends."""
        path = tmp_path / "gpu.jsonl"
        path.write_text(_sample("t0", "0", _quiet(2), phase="baseline")
                        + "\n{\"t\":\"t1\",")
        assert isolation.audit(isolation.read(str(path)))["samples"] == 1

    def test_no_samples_is_not_a_pass(self, tmp_path, capsys):
        """An unwatched run must not read the same as a clean one."""
        path = tmp_path / "gpu.jsonl"
        path.write_text("")
        assert isolation.main([str(path)]) == 1
        out = capsys.readouterr().out
        assert "UNWATCHED" in out and "nothing watched this run" in out
