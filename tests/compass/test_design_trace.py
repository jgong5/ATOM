# SPDX-License-Identifier: MIT
"""`scripts/compass/design_trace.py`: recipe C's designed workload as the
weka traces `client.sh --traces` sends, and the plan its scripts read."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "design_trace.py"


def _module():
    spec = importlib.util.spec_from_file_location("design_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _made(out: Path, *args):
    dt = _module()
    dt.main([str(out), *args])
    plan = json.loads((out / "plan.json").read_text())
    segments = {
        name: [json.loads(x) for x in (out / f"{name}.jsonl").read_text().splitlines()]
        for name in plan["segments"]
    }
    return dt, segments, plan


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    return _made(tmp_path_factory.mktemp("recipe_c"))


def test_every_trace_is_a_local_weka_trace_of_whole_blocks(made):
    _, segments, _ = made
    traces = [t for s in segments.values() for t in s]
    assert len({t["id"] for t in traces}) == len(traces)
    for t in traces:
        assert set(t) == {"id", "models", "block_size", "hash_id_scope", "requests"}
        assert (t["block_size"], t["hash_id_scope"]) == (64, "local")
        for r in t["requests"]:
            assert set(r) == {"t", "type", "model", "in", "out", "hash_ids"}
            assert r["type"] == "s" and r["model"] in t["models"] and r["out"] >= 1
            assert r["in"] % 64 == 0 and r["hash_ids"] == list(range(r["in"] // 64))


def test_a_follow_up_turn_extends_the_previous_prompt_by_its_output(made):
    _, segments, _ = made
    followed = 0
    for t in (t for s in segments.values() for t in s):
        for prev, r in zip(t["requests"], t["requests"][1:]):
            assert r["hash_ids"][: len(prev["hash_ids"])] == prev["hash_ids"]
            assert r["in"] >= prev["in"] + 64 * math.ceil(prev["out"] / 64) + 64
            assert r["t"] > prev["t"]
            followed += 1
    assert followed > 500


def test_s0_and_s1_are_the_recipe_s(made):
    _, segments, plan = made
    assert [(len(t["requests"]), t["requests"][0]["in"]) for t in segments["s0"]] == [
        (1, 100032)
    ] * 4
    s1 = segments["s1"]
    assert [t["requests"][0]["in"] // 1024 for t in s1] == [2, 16, 64, 200, 400, 700]
    for t in s1:
        assert len(t["requests"]) == 8
        assert {
            round(b["t"] - a["t"], 3) for a, b in zip(t["requests"], t["requests"][1:])
        } == {1.0}
        assert {r["out"] for r in t["requests"]} == {32, 512}
    assert plan["segments"]["s1"] == {
        "traces": 6,
        "concurrency": 1,
        "requests": 48,
        "prompt_tokens": sum(r["in"] for t in s1 for r in t["requests"]),
    }


def test_the_ladder_holds_every_rung_and_ends_a_group_a_row(made):
    dt, segments, plan = made
    rows = 56
    lanes = [t for t in segments["s2"] if t["id"].startswith("rc-s2a")]
    assert len(lanes) == 4 * rows
    groups = [lanes[i : i + 4] for i in range(0, len(lanes), 4)]
    outs = [[m["requests"][0]["out"] for m in g] for g in groups]
    for g, out in zip(groups[:-1], outs):
        assert len(set(out)) == 1
    # Members smallest first, each its share of the group's prompt, under the cap.
    for g, base in zip(groups, dt.prompts(rows)):
        sizes = [m["requests"][0]["in"] for m in g]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1] and max(sizes) <= dt.CAP
        assert sizes == [
            min(dt.CAP, 64 * round(base * f / 64)) for f in (0.55, 0.8, 1.2, 1.45)
        ]
    firsts = [o[0] for o in outs]
    assert firsts == sorted(firsts) and len(set(firsts)) == rows
    assert outs[-1] == sorted(set(outs[-1])) and outs[-1][0] == firsts[-1]
    ends = plan["group_end_steps"]
    assert firsts == [1 + math.ceil(dt.ACCEPT * e) for e in ends]
    steps = {int(r): n for r, n in plan["decode_steps"].items()}
    assert set(steps) == set(dt.RUNGS) and min(steps.values()) >= 300
    # Each row count n holds its rung, the smallest capture size at least n.
    assert ends[-1] == pytest.approx(sum(steps.values()), abs=1)


def test_the_prompt_mix_spans_its_bands(made):
    dt, _, _ = made
    first = dt.prompts(56)
    k = 1024
    assert 8 * k <= min(first) and max(first) <= 600 * k
    bands = [
        sum(lo <= x < hi for x in first)
        for lo, hi in ((0, 64 * k), (64 * k, 256 * k), (256 * k, 10**9))
    ]
    assert bands == [22, 23, 11]
    # Any eight consecutive groups in the end order reach every band.
    for i in range(len(first) - 8):
        window = first[i : i + 8]
        assert min(window) < 64 * k and max(window) >= 256 * k


def test_probe_lanes_join_every_even_wave_and_one_group_s_odd_ones(made):
    _, segments, plan = made
    probes = [t for t in segments["s2"] if t["id"].startswith("rc-s2b")]
    assert len(probes) == 32
    waves = plan["waves"]
    for g in range(8):
        members = probes[4 * g : 4 * g + 4]
        assert len({m["requests"][0]["in"] for m in members}) == 1
        turns = [len(m["requests"]) - 1 for m in members]
        assert all(n >= (waves + 1) // 2 for n in turns)
        assert turns == sorted(turns, reverse=True)
    assert {p["requests"][0]["in"] // 1024 for p in probes} == {16, 96, 256, 512}
    gaps = [
        b["t"] - a["t"]
        for p in probes
        for a, b in zip(p["requests"], p["requests"][1:])
    ]
    assert min(gaps) >= 0.5 and max(gaps) <= 2 * plan["period"]


def test_the_plan_counts_what_is_written_and_a_second_run_writes_the_same(
    made, tmp_path
):
    _, segments, plan = made
    for name, s in segments.items():
        p = plan["segments"][name]
        assert (p["traces"], p["requests"]) == (
            len(s),
            sum(len(t["requests"]) for t in s),
        )
    assert plan["segments"]["s2"]["concurrency"] == len(segments["s2"])
    _, again, _ = _made(tmp_path)
    assert again == segments


def test_step_times_and_rows_reshape_the_ladder(tmp_path):
    _, segments, plan = _made(
        tmp_path, "--rows", "52", "--step-ms", '{"64": 80}', "--waves", "4"
    )
    assert sum("s2a" in t["id"] for t in segments["s2"]) == 4 * 52
    assert plan["decode_steps"]["64"] == 500 and plan["waves"] == 4


def test_s3_holds_eight_sessions_a_rank_of_growing_context(made):
    _, segments, plan = made
    assert list(plan["segments"]) == ["s0", "s1", "s2", "s3"]
    s3 = segments["s3"]
    assert plan["segments"]["s3"]["concurrency"] == len(s3) == 32
    first = [t["requests"][0]["in"] for t in s3]
    assert first == sorted(first) and first[0] // 1024 == 8 and first[-1] // 1024 < 256
    assert {len(t["requests"]) for t in s3} == {40}
    gaps = {
        round(b["t"] - a["t"], 3)
        for t in s3
        for a, b in zip(t["requests"], t["requests"][1:])
    }
    assert gaps == {0.5, 1.0, 1.5, 3.0}


def test_no_s3_leaves_it_out(tmp_path):
    _, segments, plan = _made(tmp_path, "--s3", "0")
    assert list(plan["segments"]) == list(segments) == ["s0", "s1", "s2"]
