"""Merging a sharded calibration, and saying what it is not.

A long calibration collected as several fresh processes replaces a monolithic
sweep that died in a device fault. The merged table has to be usable *and* has
to carry the fact that it was not collected the way a sweep is, or a later
reader takes it for evidence that the fault went away.
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


merge_sweep = _load("merge_sweep")


def _shard(root: Path, index: int, rows, rounds):
    (root / f"shard{index}.rounds.json").write_text(
        json.dumps({"shard": f"{index}/2", "rounds": rounds}))
    (root / f"shard{index}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


class TestTheMergedTable:
    def test_rows_are_labelled_with_the_pass_they_came_from(self, tmp_path):
        """Warmup and steady are not separable after the fact by anything the
        engine records, so they are attributed by wall-clock interval."""
        _shard(tmp_path, 0,
               [{"seconds": 1.0, "started_at": 100.5},
                {"seconds": 0.4, "started_at": 200.5}],
               [{"round": 0, "pass": "warmup", "t0": 100.0, "t1": 101.0},
                {"round": 1, "pass": "steady", "t0": 200.0, "t1": 201.0}])
        out = tmp_path / "merged.jsonl"
        assert merge_sweep.main([str(tmp_path), "--out", str(out)]) == 0
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r["sweep_pass"] for r in rows] == ["warmup", "steady"]
        assert [r["sweep_round"] for r in rows] == [0, 1]

    def test_a_row_outside_every_round_is_named_not_dropped(self, tmp_path):
        """Startup and teardown steps belong to no round; losing them
        silently would make the row count look like the round count."""
        _shard(tmp_path, 0, [{"seconds": 9.0, "started_at": 1.0}],
               [{"round": 0, "pass": "warmup", "t0": 100.0, "t1": 101.0}])
        out = tmp_path / "merged.jsonl"
        merge_sweep.main([str(tmp_path), "--out", str(out)])
        row = json.loads(out.read_text().splitlines()[0])
        assert row["sweep_pass"] is None
        manifest = json.loads((tmp_path / "merged.jsonl.manifest.json")
                              .read_text())
        assert manifest["shards"][0]["passes"]["unattributed"] == 1

    def test_a_torn_final_line_costs_one_step_not_the_shard(self, tmp_path):
        (tmp_path / "shard0.rounds.json").write_text(
            json.dumps({"shard": "0/1", "rounds": []}))
        (tmp_path / "shard0.jsonl").write_text(
            json.dumps({"seconds": 1.0, "started_at": 1.0}) + "\n{\"sec")
        out = tmp_path / "merged.jsonl"
        merge_sweep.main([str(tmp_path), "--out", str(out)])
        assert len(out.read_text().splitlines()) == 1


class TestWhatTheManifestHasToSay:
    def test_the_limitation_is_recorded_not_left_to_the_reader(self, tmp_path):
        _shard(tmp_path, 0, [{"seconds": 1.0, "started_at": 100.5}],
               [{"round": 0, "pass": "warmup", "t0": 100.0, "t1": 101.0}])
        out = tmp_path / "merged.jsonl"
        merge_sweep.main([str(tmp_path), "--out", str(out)])
        manifest = json.loads((tmp_path / "merged.jsonl.manifest.json")
                              .read_text())
        assert "not as one sweep" in manifest["limitation"]
        assert "device fault" in manifest["limitation"]

    def test_a_failed_shard_is_missing_and_says_so(self, tmp_path):
        """Its rows were renamed away by the driver. An incomplete table must
        not merge quietly into something that reads as complete."""
        _shard(tmp_path, 0, [{"seconds": 1.0, "started_at": 100.5}],
               [{"round": 0, "pass": "warmup", "t0": 100.0, "t1": 101.0}])
        (tmp_path / "shard1.rounds.json").write_text(
            json.dumps({"shard": "1/2", "rounds": []}))
        out = tmp_path / "merged.jsonl"
        assert merge_sweep.main([str(tmp_path), "--out", str(out)]) == 1
        manifest = json.loads((tmp_path / "merged.jsonl.manifest.json")
                              .read_text())
        assert manifest["missing"] == ["shard1"]

    def test_per_rank_tables_are_selectable(self, tmp_path):
        (tmp_path / "shard0.rounds.json").write_text(
            json.dumps({"shard": "0/1", "rounds": []}))
        (tmp_path / "shard0.tp0.jsonl").write_text(
            json.dumps({"seconds": 2.0, "started_at": 1.0}) + "\n")
        out = tmp_path / "merged.jsonl"
        assert merge_sweep.main([str(tmp_path), "--out", str(out),
                                 "--rank", "tp0"]) == 0
        assert len(out.read_text().splitlines()) == 1
