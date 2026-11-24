# SPDX-License-Identifier: MIT
"""The unified calibration procedure v1.1's design tooling (#739 sections 2-4):
`atom.compass.calib` and scripts/compass/{gen,plan,check}_v1.py, cal_v1.sh and
dry_v1.sh, on synthetic configs and laws."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.calib import check, columns, deploy, design, engine, ident, report

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "compass"
CONFIGS = SCRIPTS / "calib"

DEPLOYMENT = {
    "dp": 2,
    "rungs": [1, 2, 4, 8],
    "max_num_seqs": 8,
    "max_num_batched_tokens": 4096,
    "block_size": 128,
    "max_model_len": 65536,
    "kv_tokens": 400000,
    "drafts": 1,
    "acceptance": 1.5,
    "prefill_buckets": [256, 512, 1024],
    "moe_routing": [16, 2],
}


def _config(**over) -> dict:
    raw = {
        "name": "tiny",
        "deployment": dict(DEPLOYMENT),
        "columns": {
            "decode": ["step", "context", "graph_padding", "row_padding"],
            "decode_variants": [[], ["peer_tokens"]],
            "prefill": [
                "step",
                "tokens",
                "query_square",
                "query_cached",
                "history",
                "moe_rows",
            ],
        },
        "noise": {"decode": 0.005, "prefill": 0.004},
        "acceptance": {"rung_steps": 12},
        "range": {"min_context": 256, "min_prompt": 256, "min_seq_tokens": 64},
        "timing": {"server_start_s": 10, "segment_overhead_s": 5},
        "segments": [
            {
                "name": "D-a",
                "kind": "drain",
                "lanes": [{"pin": 0, "groups": [{"rows": 8, "prompt": 1024}]}],
            },
            {
                "name": "D-b",
                "kind": "drain",
                "lanes": [
                    {
                        "pin": 0,
                        "groups": [
                            {"rows": 7, "prompt": 512, "then": 128},
                            {"rows": 1, "prompt": 20000, "last": True},
                        ],
                    },
                    {
                        "name": "peer",
                        "pin": 1,
                        "hold": True,
                        "groups": [{"rows": 1, "prompt": 30000}],
                    },
                ],
            },
            {
                "name": "D-c",
                "kind": "drain",
                "lanes": [
                    {"pin": None, "groups": [{"rows": 16, "prompt": [512, 8192]}]}
                ],
            },
            {
                "name": "P-q",
                "kind": "quads",
                "scales": [512, 2048, 4096],
                "shares": [0.0, 1.0],
                "sparse_steps": [0, 20],
            },
            {
                "name": "P-t",
                "kind": "trickle",
                "firsts": [8192, 16384],
                "deltas": [1024, 2048],
                "outs": [1, 4],
                "gaps": [0.1],
                "turns": 4,
            },
            {
                "name": "P-r",
                "kind": "rows",
                "requests": 64,
                "prompt": [64, 1024],
                "concurrency": 16,
            },
        ],
    }
    for k, v in over.items():
        raw[k] = v
    return raw


def _vector(features, **extra) -> dict:
    return {
        "coefficients": features,
        "hull": {"equal": {f: 0.0 for f in features if f != "step"}, "facets": []},
        **extra,
    }


def _law(rungs, dp: int) -> dict:
    """A step linear law priced like a small DP model, every hull a point."""
    decode = {
        str(r): _vector(
            {"step": 0.004 + 0.0002 * r, "context": 2e-9, "graph_padding": 5e-10}
        )
        for r in rungs
    }
    prefill = {
        "step": 0.01,
        "tokens": 6e-6,
        "query_square": 2e-11,
        "query_cached": 1e-11,
        "history": 5e-9,
    }
    if dp > 1:
        prefill["moe_rows"] = 2e-6
    return {"provenance": "synthetic", "prefill": _vector(prefill), "decode": decode}


HOST = {
    "params": {
        "laws": {
            "decode": {"E": [0.001, 0.0, 0.0], "W_launch": [0.001, 0.0, 0.0]},
            "prefill": {"W_launch": [0.02, 0.0, 0.0001]},
            "middle": {"W_launch": [0.005, 0.0, 0.0001]},
        }
    }
}


@pytest.fixture(scope="module")
def tiny():
    cfg = deploy.parse(_config())
    law = engine.Law(
        _law(cfg.deployment.rungs, cfg.deployment.dp), HOST, cfg.deployment.dp
    )
    segments = design.generate(cfg)
    return cfg, law, segments, engine.rehearse(cfg, segments, law)


# -- the design ---------------------------------------------------------------


def test_the_common_departure_schedule_holds_every_rung_its_steps():
    cfg = deploy.parse(_config())
    dep = cfg.deployment
    leave = design.departures(8, dep, 12)
    held = {}
    for n, (before, at) in enumerate(zip([0] + leave, leave)):
        held[dep.rung(8 - n)] = held.get(dep.rung(8 - n), 0) + at - before
    assert all(held[r] >= 12 for r in dep.rungs)
    # a row leaves at exactly its step at the forced acceptance
    for steps in (1, 7, 60, 333):
        out = design.out_for(steps, dep)
        assert math.ceil((out - 1) / dep.per_step) == steps


def test_a_design_is_weka_traces_whose_later_turns_extend_the_history(tmp_path):
    cfg = deploy.parse(_config())
    doc = design.write(design.generate(cfg), cfg, tmp_path)
    ids = []
    for seg in doc["segments"]:
        for lane in seg["lanes"]:
            traces = [
                json.loads(x)
                for x in (tmp_path / lane["file"]).read_text().splitlines()
            ]
            assert len(traces) == lane["traces"]
            for t in traces:
                ids.append(t["id"])
                assert t["id"].startswith(f"{seg['name']}.{lane['name']}.")
                assert set(t) == {
                    "id",
                    "models",
                    "block_size",
                    "hash_id_scope",
                    "requests",
                }
                for prev, r in zip(t["requests"], t["requests"][1:]):
                    assert r["hash_ids"][: len(prev["hash_ids"])] == prev["hash_ids"]
                    assert r["t"] >= prev["t"] and r["in"] > prev["in"]
                for r in t["requests"]:
                    assert (
                        r["in"] % 64 == 0
                        and r["in"] + r["out"] <= cfg.deployment.max_model_len
                    )
    assert len(set(ids)) == len(ids)
    pins = {
        (s["name"], x["name"]): x["pin"] for s in doc["segments"] for x in s["lanes"]
    }
    assert pins[("D-b", "peer")] == 1 and pins[("D-c", "l0")] is None
    back = design.read(tmp_path, cfg)
    assert [s.name for s in back] == [s["name"] for s in doc["segments"]]


def test_a_held_lane_stays_to_the_end_and_anchors_leave_last():
    cfg = deploy.parse(_config())
    seg = next(s for s in design.generate(cfg) if s.name == "D-b")
    busy, _ = seg.lanes
    outs = [s.turns[-1].out for s in busy.sessions]
    assert outs[-1] == max(outs)  # the anchor, listed `last`, leaves last
    # the peer holds its row until the busy rank's anchor leaves: the same group
    # step, each counted from the end of its own rank's prefill
    plan, chunks = seg.plan["lanes"], seg.plan["prefill_chunks"]
    assert (
        max(plan["l0"]["leave_steps"]) + chunks[0]
        == plan["peer"]["leave_steps"][0] + chunks[1]
    )
    # the `then` group: one token, then a short turn on its history
    first = busy.sessions[0]
    assert first.turns[0].out == 1 and len(first.turns) == 2
    # the busy rank finishes its prefill first: its schedule lags the peer's chunks
    assert seg.plan["lanes"]["l0"]["lag_steps"] > 0


def test_every_config_in_the_tree_generates(tmp_path):
    for path in sorted(CONFIGS.glob("*.json")):
        cfg = deploy.load(path)
        segments = design.generate(cfg)
        assert segments and all(s.lanes for s in segments)
        for seg in segments:
            for lane in seg.lanes:
                assert lane.pin is None or 0 <= lane.pin < cfg.deployment.dp


# -- the rehearsal ------------------------------------------------------------


def test_the_rehearsal_drains_each_pinned_segment_through_every_rung(tiny):
    cfg, _, _, got = tiny
    rows = {r["segment"]: r for r in report.segment_rows(got.steps, got.runs)}
    for name in ("D-a", "D-c"):
        assert all(
            rows[name]["decode_rungs"].get(str(r), 0) >= 12
            for r in cfg.deployment.rungs
        )
    # chained by events: each segment starts after the last one ended, plus the overhead
    for a, b in zip(got.runs, got.runs[1:]):
        assert b.start == pytest.approx(a.end + cfg.timing["segment_overhead_s"])
    assert got.gpu_minutes == pytest.approx(got.wall_s / 60)


def test_a_pinned_drain_decodes_on_its_rank_beside_dummy_peers(tiny):
    _, _, _, got = tiny
    steps = [s for s in got.steps if s.seg == "D-a" and s.kind == "decode"]
    assert steps and all([k for k, *_ in s.ranks] == [0] for s in steps)
    s = steps[0]
    assert s.view.group_dummies == 1 and s.rung == 8
    # the group's MoE tokens are its real rows': a dummy adds none, as records state
    assert s.view.moe_tokens == 8 * 2 and s.view.moe_rows == 8 * 2 * 2


def test_unpinned_sessions_started_together_land_one_a_rank(tiny):
    _, _, _, got = tiny
    quads = [s for s in got.steps if s.seg == "P-q" and s.kind == "prefill"]
    first = quads[0]
    assert sorted(k for k, v, *_ in first.ranks if not v.dp_dummy) == [0, 1]


def test_a_peer_holding_the_rung_prices_a_rank_below_it(tiny):
    cfg, _, _, got = tiny
    held = [
        s
        for s in got.steps
        if s.seg == "D-b"
        and s.kind == "decode"
        and report.combo(s, cfg.deployment).split()[2] == "peer"
    ]
    assert held  # the peer's 30k-token row outprices the busy rank's short rows


def test_the_second_turns_prefill_together_on_deep_histories(tiny):
    _, _, _, got = tiny
    deep = [
        view
        for s in got.steps
        if s.seg == "D-b" and s.kind == "prefill"
        for _, view, *_ in s.ranks
        if sum(1 for r in view.requests if r.cached_tokens) >= 4
    ]
    assert deep


# -- columns and identifiability --------------------------------------------------


def test_a_backend_column_is_counted_by_the_backend_and_an_opt_in_one_provisionally():
    dep = deploy.parse(_config()).deployment
    view = BatchView(
        (RequestShape(2, 1000, True), RequestShape(2, 3000, True)),
        capture_rung=4,
        moe_rows=4 * 2 * 2,
        moe_tokens=10,
        produces_output=True,
        dp_rank=0,
    )
    got = columns.values(
        view,
        "decode",
        ["step", "context", "graph_padding", "row_padding", "peer_tokens"],
        dep,
    )
    assert got == (1.0, 4000.0, view.graph_padding, 2 * 3000 - 4000, 10 - 4)
    assert columns.backend_prices("decode", "context")
    with pytest.raises(ValueError, match="neither"):
        columns.values(view, "decode", ["step", "no_such_column"], dep)


def test_vif_is_one_apart_and_infinite_for_a_copy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    X = np.column_stack([np.ones(200), a, b])
    assert all(v < 1.1 for v in ident.vif(X))
    assert ident.vif(np.column_stack([np.ones(200), a, 2 * a + 1]))[1] == math.inf


def test_g_is_the_worst_relative_standard_error_over_the_vertices():
    X = np.column_stack([np.ones(4), [1.0, 2.0, 3.0, 4.0]])
    y = np.array([2.0, 3.0, 4.0, 5.0])  # beta (1, 1)
    V, vy = np.array([[1.0, 10.0]]), np.array([11.0])
    got = ident.assess(["step", "x"], X, y, V, vy, sigma=0.01, z=2.0)
    Xw = X / y[:, None]
    want = 2.0 * 0.01 * math.sqrt(V[0] @ np.linalg.inv(Xw.T @ Xw) @ V[0]) / 11.0
    assert got["rank"] == 2 and got["g"] == pytest.approx(want)


def test_the_tiny_design_is_assessed_per_vector_and_variant(tiny):
    cfg, law, _, got = tiny
    vectors = ident.vectors(cfg, got.steps, law)
    names = [v["vector"] for v in vectors]
    assert names == ["prefill"] + [f"decode rung {r}" for r in cfg.deployment.rungs]
    rung1 = vectors[1]["variants"][0]
    assert {"graph_padding", "row_padding"} <= set(
        rung1["structural"]
    )  # one row: no padding
    for v in vectors[1:]:
        assert [x["variant"] for x in v["variants"]] == ["base", "peer_tokens"]
    text = "\n".join(report.render_vectors(vectors))
    assert "decode rung 8" in text and "provisional: row_padding" in text


def test_one_dp_rank_keeps_both_paddings_and_is_judged_by_g():
    raw = _config()
    raw["deployment"] = dict(DEPLOYMENT, dp=1, rungs=[1, 2, 3, 4, 8])
    raw["columns"]["prefill"] = [
        "step",
        "tokens",
        "query_square",
        "query_cached",
        "history",
    ]
    raw["segments"] = [
        {
            "name": "D-a",
            "kind": "drain",
            "lanes": [{"pin": 0, "groups": [{"rows": 8, "prompt": 1024}]}],
        },
        {
            "name": "D-b",
            "kind": "drain",
            "lanes": [
                {
                    "pin": 0,
                    "groups": [
                        {"rows": 7, "prompt": 512},
                        {"rows": 1, "prompt": 30000, "last": True},
                    ],
                }
            ],
        },
        {
            "name": "P-t",
            "kind": "trickle",
            "firsts": [8192, 16384, 24576],
            "deltas": [1024, 3072],
            "outs": [1, 4],
            "gaps": [0.1],
            "turns": 5,
        },
        {
            "name": "P-r",
            "kind": "rows",
            "requests": 64,
            "prompt": [64, 2048],
            "concurrency": 8,
        },
    ]
    cfg = deploy.parse(raw)
    law = engine.Law(_law(cfg.deployment.rungs, 1), HOST, 1)
    got = engine.rehearse(cfg, design.generate(cfg), law)
    vectors = {v["vector"]: v for v in ident.vectors(cfg, got.steps, law)}
    rung2 = vectors["decode rung 2"]["variants"][0]
    assert (
        rung2["rank"] == len(rung2["columns"]) - 1
    )  # graph_padding == row_padding at 2 rows
    assert (
        rung2["g"] < math.inf
        and rung2["rule6"]
        and "rank" not in " ".join(rung2["fails"])
    )


# -- the scripts ----------------------------------------------------------------


def _run(*argv, **kw):
    path = f"{Path(sys.executable).parent}:/usr/bin:/bin"  # the python3 the scripts run is this one
    env = {"PYTHONPATH": str(ROOT), "PATH": path, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [*argv], check=False, capture_output=True, text=True, env=env, **kw
    )


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    base = tmp_path_factory.mktemp("v1")
    (base / "cfg.json").write_text(json.dumps(_config()))
    law_dir = base / "law"
    (law_dir / "host").mkdir(parents=True)
    (law_dir / "law.json").write_text(json.dumps(_law(DEPLOYMENT["rungs"], 2)))
    (law_dir / "host" / "host_laws.json").write_text(json.dumps(HOST))
    out = base / "dry"
    got = _run(
        "bash",
        str(SCRIPTS / "dry_v1.sh"),
        str(base / "cfg.json"),
        str(law_dir),
        str(out),
        cwd="/",
    )
    return base, out, got


def test_dry_v1_plans_and_judges_a_design(planned):
    _, out, got = planned
    assert got.returncode in (0, 1), got.stderr
    plan = json.loads((out / "design" / "plan.json").read_text())
    assert [s["segment"] for s in plan["sequence"]][:2] == ["D-a", "D-b"]
    assert plan["sequence"][0]["after"] == "server start"
    assert plan["sequence"][1]["after"].startswith("D-a: every lane")
    assert plan["server_env"]["ATOM_DP_LB_REQ_EQUIV"] == "1000000000"
    lanes = plan["sequence"][1]["lanes"]
    assert [x["pin"] for x in lanes] == [0, 1] and len({x["seed"] for x in lanes}) == 2
    r = plan["rehearsal"]
    assert r["gpu_minutes"] > 0 and r["combos"]["D-a"]
    assert got.returncode == (0 if r["pass"] else 1)
    assert "## Identifiability" in (out / "design" / "plan.md").read_text()


def test_dry_v1_refuses_an_existing_out(planned):
    base, out, _ = planned
    again = _run(
        "bash",
        str(SCRIPTS / "dry_v1.sh"),
        str(base / "cfg.json"),
        str(base / "law"),
        str(out),
        cwd="/",
    )
    assert again.returncode == 2 and "exists" in again.stderr


def test_cal_v1_refuses_a_design_without_a_plan(tmp_path):
    got = _run(
        "bash",
        str(SCRIPTS / "cal_v1.sh"),
        str(tmp_path),
        str(tmp_path / "out"),
        "record",
        "--",
        cwd="/",
    )
    assert got.returncode == 2 and "plan.json" in got.stderr


def _record(tmp_path, cfg, design_doc):
    """A two-rank record of one decode segment: rank 0 decodes two requests at
    rung 2 for 30 steps while rank 1 runs dummies, as `cal_v1.sh` lays it out."""
    seg = design_doc["segments"][0]
    lane = seg["lanes"][0]
    steps = tmp_path / "rec" / "real" / "steps"
    steps.mkdir(parents=True)
    names = ["a", "b"]
    lines = {0: [], 1: []}
    t = 0
    for n in range(30):
        for r in (0, 1):
            line = {
                "step": n,
                "dp_rank": r,
                "is_dummy_run": r == 1,
                "named": [[k, k] for k in names] if n == 0 and r == 0 else [],
                "batch": [[k] for k in names] if r == 0 else [],
                "rows": (
                    [[2, 1000 + 3 * n, True], [2, 5000 + 3 * n, True]]
                    if r == 0
                    else [[1, 1, True]]
                ),
                "rung": 2,
                "moe_rows": 2 * 2 * 2,
                "moe_tokens": 6,
                "produces_output": True,
                "t_enter_ns": t,
                "t_exit_ns": t + 9_000_000,
                "d_target_ns": 8_000_000,
            }
            lines[r].append(line)
        t += 10_000_000
    for r, got in lines.items():
        (steps / f"dp{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in got))
    export = tmp_path / "rec" / seg["name"] / lane["name"] / "artifacts"
    export.mkdir(parents=True)
    trace = f"{seg['name']}.{lane['name']}.x"
    (export / "client_run.json").write_text(
        json.dumps(
            {
                "requests": {
                    k: {"trace": trace, "sampling": {"ignore_eos": True}} for k in names
                }
            }
        )
    )
    return tmp_path / "rec"


def test_check_reads_a_record_per_segment_and_compares_it_with_the_plan(tmp_path):
    cfg = deploy.parse(_config())
    doc = design.write(design.generate(cfg), cfg, tmp_path / "design")
    law = engine.Law(_law(cfg.deployment.rungs, 2), HOST, 2)
    rec = _record(tmp_path, cfg, doc)
    steps, measured = check.record_steps(rec, doc, cfg, law)
    assert len(steps) == 30 and {s.seg for s in steps} == {"D-a"}
    assert all(s.kind == "decode" and s.rung == 2 and s.gpu_bound for s in steps)
    assert measured[0] == pytest.approx(0.008)
    actual = report.combos(steps, cfg.deployment)
    assert actual == {"D-a": {"decode r2 own 1rk": 30}}
    rows = check.compare(
        {"D-a": {"decode r2 own 1rk": 60, "decode r8 own 1rk": 60}}, actual, floor=20
    )
    short = {r["combo"]: r for r in rows}
    assert (
        short["decode r2 own 1rk"]["ratio"] == 0.5
        and not short["decode r2 own 1rk"]["short"]
    )
    assert short["decode r8 own 1rk"]["short"]


def test_check_v1_writes_its_report(tmp_path):
    cfg_raw = _config()
    cfg = deploy.parse(cfg_raw)
    doc = design.write(design.generate(cfg), cfg, tmp_path / "design")
    (tmp_path / "design" / "plan.json").write_text(
        json.dumps({"rehearsal": {"combos": {"D-a": {"decode r2 own 1rk": 30}}}})
    )
    law_path = tmp_path / "law.json"
    law_path.write_text(json.dumps(_law(cfg.deployment.rungs, 2)))
    rec = _record(tmp_path, cfg, doc)
    got = _run(
        sys.executable,
        str(SCRIPTS / "check_v1.py"),
        str(tmp_path / "design"),
        str(rec),
        "--law",
        str(law_path),
        cwd="/",
    )
    assert got.returncode in (0, 1), got.stderr
    out = json.loads((rec / "check.json").read_text())
    assert out["steps"] == 30 and out["combos"][0]["actual"] == 30
    assert "## Identifiability on the record" in (rec / "check.md").read_text()


def test_a_config_states_its_intercept_first():
    raw = copy.deepcopy(_config())
    raw["columns"]["decode"] = ["context", "step"]
    with pytest.raises(ValueError, match="intercept"):
        deploy.parse(raw)
