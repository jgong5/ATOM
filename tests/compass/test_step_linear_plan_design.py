# SPDX-License-Identifier: MIT
"""A designed-load calibration's decode design plan and prefill point plan,
each planned from a served real record: its engine files, its step record,
its request bodies and the calibration traces its client run was given."""

import json
import math

import numpy as np
import pytest

from atom.compass.backends.step_linear import EAGER
from atom.compass.backends.step_linear_plan import design, main, trace_design

BLOCK = 16


def _engine(**fields):
    return {
        "ladder": [1, 2, 4, 8],
        "kv_cache_block_size": BLOCK,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 64,
        "max_model_len": 1 << 20,
        "num_kvcache_blocks": 1 << 20,
        "prefill_buckets": [],
    } | fields


def _record(
    path,
    contexts,
    histories=((),),
    engine=None,
    traces=(10**6,),
    model="qwen3",
    atom_args=(),
    advances=(1,),
    body=4,
):
    """A real record at `path`, its DP rank k the k-th of `contexts`: a request
    prefilled whole and decoded twice, its decode rows at a context and that
    plus the next of `advances`, for each of `contexts[k]`; one prefilled on a
    cached history for each of `histories[k]`; each request's body `body`
    bytes a prompt token. Its client run was given a trace of each longest
    prompt in `traces`, and its server ran `model` under `atom_args`."""
    steps = path / "steps"
    steps.mkdir(parents=True)
    stamps, n = [], 0
    for rank, own in enumerate(contexts):
        lines = []
        mine = histories[rank] if rank < len(histories) else ()
        for i, (prompt, history) in enumerate(
            [(c, None) for c in own] + [(h + 16, h) for h in mine]
        ):
            rid, n = n, n + 1
            q = prompt if history is None else 16
            lines.append(
                {
                    "is_dummy_run": False,
                    "batch": [[rid, q, prompt]],
                    "rows": [[q, prompt, False]],
                    "named": [[rid, f"key{rid}", f"chatcmpl-{rid}"]],
                }
            )
            stamps.append(f"{2 * rid + 1} 1 body {2 * rid} {body * prompt} 0 0")
            if history is None:
                step = advances[i % len(advances)]
                for kv in (prompt, prompt + step):
                    lines.append(
                        {
                            "is_dummy_run": False,
                            "batch": [[rid, 1, kv]],
                            "rows": [[1, kv, True]],
                            "named": [],
                        }
                    )
        text = "".join(json.dumps({"step": s, **x}) + "\n" for s, x in enumerate(lines))
        (steps / f"dp{rank}.jsonl").write_text(text)
        stated = engine or _engine()
        (steps / f"dp{rank}.engine.json").write_text(json.dumps(stated))
    (steps / "intake.stamps").write_text("\n".join(stamps) + "\n")
    given = path / "client" / "traces"
    given.mkdir(parents=True)
    for k, longest in enumerate(traces):
        trace = {"id": f"t{k}", "requests": [{"in": 10, "out": 2}, {"in": longest}]}
        (given / f"{k:05d}.json").write_text(json.dumps(trace))
    weights = path / "model"
    weights.mkdir()
    (weights / "config.json").write_text(json.dumps({"model_type": model}))
    argv = ["-c", "record", "--out", str(path), "--", "--model", str(weights)]
    (path / "run.json").write_text(json.dumps({"argv": argv + list(atom_args)}))
    return path


def _spread(lo, hi, count=200):
    return [int(x) for x in np.linspace(lo, hi, count)]


def test_each_rung_runs_its_level_groups_then_its_top_rows_each_gap_batch_after(
    tmp_path,
):
    # Rungs 1, 2, 4 and 8 captured and the eager rung at the bound, 16:
    # batches of 3, 5 and 9 rows replay the rungs above 2.
    plan = design(_record(tmp_path, [_spread(1000, 9000)]), 1)
    rungs = list(dict.fromkeys(g["rung"] for g in plan["groups"]))
    assert rungs == [1, 2, 4, 8, EAGER]
    at = [g for g in plan["groups"] if g["rung"] == EAGER]
    kinds = [(g["kind"], g["gap"], len(g["rows"][0])) for g in at]
    assert kinds == [
        ("bottom", False, 16),
        ("bottom", True, 9),
        ("middle", False, 16),
        ("middle", True, 9),
        ("top", False, 16),
        ("top", True, 9),
        ("one_long", False, 16),
        ("one_long", True, 9),
        ("half_alternating", False, 16),
        ("half_halves", False, 16),
    ]
    # Below four rows no half-long pair, and one rank no DP group.
    assert [g["kind"] for g in plan["groups"] if g["rung"] == 2] == [
        "bottom",
        "middle",
        "top",
        "one_long",
    ]
    cuts = plan["sessions"][0]
    by_kind = {g["kind"]: g["rows"][0] for g in at if not g["gap"]}
    # A group's rows are the rung's first sessions, one a row.
    assert [i for i, _ in by_kind["middle"]] == list(range(16))
    assert [t for _, t in by_kind["middle"]] == [s["cuts"][1] for s in cuts[:16]]
    assert by_kind["one_long"] == [[0, cuts[0]["cuts"][2]]] + [
        [i, cuts[i]["cuts"][0]] for i in range(1, 16)
    ]
    # The half-long pair: one multiset in two orders, sent on one fill.
    tops = [[i, cuts[i]["cuts"][2]] for i in range(8)]
    bottoms = [[i, cuts[i]["cuts"][0]] for i in range(8, 16)]
    assert by_kind["half_halves"] == tops + bottoms
    assert by_kind["half_alternating"][:4] == [tops[0], bottoms[0], tops[1], bottoms[1]]
    assert [g["fill"] for g in at if g["kind"].startswith("half")] == [True, False]


def test_levels_are_the_decode_rows_percentiles_and_each_draw_sits_within_ten(
    tmp_path,
):
    contexts = _spread(1000, 100_000, 400)
    traces = (50_000, 120_000, 200_000)
    plan = design(_record(tmp_path, [contexts], traces=traces), 3)
    rows = sorted(contexts + [c + 1 for c in contexts])
    assert plan["levels"] == [list(np.percentile(rows, (10, 50, 90)))]
    longest = {f"t{k}": n for k, n in enumerate(traces)}
    for session in plan["sessions"][0]:
        for cut, level in zip(session["cuts"], (10, 50, 90)):
            assert np.percentile(rows, level - 10) <= cut
            assert cut <= np.percentile(rows, level + 10)
        # Its conversation holds its top cut.
        assert longest[session["trace"]] > session["cuts"][2]
    assert {s["trace"] for s in plan["sessions"][0]} == {"t1", "t2"}


def test_a_wide_rung_scales_its_three_levels_together_to_the_pool_cap(tmp_path):
    # A pool of 64 blocks, so a rung's top cuts hold at most 819 tokens.
    engine = _engine(num_kvcache_blocks=64, max_num_seqs=8)
    plan = design(_record(tmp_path, [_spread(100, 400)], engine=engine), 1)
    cuts = plan["sessions"][0]
    full = [g for g in plan["groups"] if not g["gap"]]
    top = {g["rung"]: g["rows"][0] for g in full if g["kind"] == "top"}
    bottom = {g["rung"]: g["rows"][0] for g in full if g["kind"] == "bottom"}
    # Two rows fit unscaled; from four, the rung's top cuts together fill the
    # cap, each row short of its share by under a token, and every level
    # scales by the same factor.
    assert [t for _, t in top[2]] == [s["cuts"][2] for s in cuts[:2]]
    for rung in 4, 8:
        sent = [t for _, t in top[rung]]
        assert 819.2 - rung < sum(sent) <= 819.2
        scale = 819.2 / sum(s["cuts"][2] for s in cuts[:rung])
        assert [t for _, t in bottom[rung]] == [
            int(scale * s["cuts"][0]) for s in cuts[:rung]
        ]


def test_under_dp_every_rank_runs_the_rung_and_the_middle_runs_beside_uneven_peers(
    tmp_path,
):
    ladder = [1, 2, 4, 8, 16, 24, 32, 40, 48, 64]
    # A pool of 1.6M tokens a rank, which every rank's rung-64 top cuts pass.
    engine = _engine(
        ladder=ladder,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        num_kvcache_blocks=100_000,
    )
    contexts = [_spread(1000 * (k + 1), 50_000) for k in range(4)]
    plan = design(_record(tmp_path, contexts, engine=engine), 2)
    # Ten rungs; eight gap batches of four groups; at four rows and more the
    # half-long pair; four DP groups at every rung.
    assert len(plan["groups"]) == 128
    dp = [g for g in plan["groups"] if g["kind"].startswith("dp_")]
    assert len(dp) == 40
    for g in plan["groups"]:
        if not g["kind"].startswith("dp_"):
            assert len({len(rows) for rows in g["rows"]}) == 1
    # Each kind's full rank moves from rung to rung, as from group to group.
    at = {
        r: [[len(rows) for rows in g["rows"]] for g in dp if g["rung"] == r]
        for r in ladder
    }
    assert at[8] == [[4, 4, 4, 8], [8, 1, 1, 1], [8, 8, 0, 8], [0, 0, 8, 0]]
    assert at[16] == [[16, 8, 8, 8], [1, 16, 1, 1], [16, 16, 16, 0], [0, 0, 0, 16]]
    # Each rank's rung-64 top cuts fill 0.8 of its own pool, scaled by its own
    # factor, each row short of its share by under a token.
    tops = [g for g in plan["groups"] if g["kind"] == "top" and not g["gap"]]
    for rows in tops[-1]["rows"]:
        assert 0.8 * 1_600_000 - 64 < sum(t for _, t in rows) <= 0.8 * 1_600_000
    # Each rank's sessions are its own rows' draws.
    assert [len(s) for s in plan["sessions"]] == [64] * 4
    assert min(s["cuts"][0] for s in plan["sessions"][0]) < 4000
    assert all(s["cuts"][0] >= 4000 for s in plan["sessions"][3])
    assert plan["rows"][0] == sum(len(g["rows"][0]) for g in plan["groups"])


@pytest.mark.parametrize(
    "model, atom_args, replay",
    [
        ("qwen3", (), 0),
        ("deepseek_v41", (), BLOCK),
        ("deepseek_v41", ("--no-encoder-swa-bounded-replay",), 0),
    ],
)
def test_a_fill_resumes_at_the_last_block_cached_and_a_replaying_hit_one_before(
    tmp_path, model, atom_args, replay
):
    # Every row at 1008 or 1009 tokens, 1008 a whole number of blocks, all of
    # which a 1008-token prompt leaves cached. A session's first fill computes
    # its prompt; a later one of 1009 tokens resumes at 1008, and one of 1008,
    # short of its own last block, at 992, a replaying hit a block before.
    record = _record(tmp_path, [[1008] * 20], model=model, atom_args=atom_args)
    plan = design(record, 1)
    assert plan["hit_replay"] == replay
    resumed, filled, computed = {1008: 992, 1009: 1008}, set(), 0
    for g in plan["groups"]:
        for i, tokens in g["rows"][0] if g["fill"] else ():
            later = i in filled
            computed += tokens - (resumed[tokens] - replay if later else 0)
            filled.add(i)
    assert len(filled) == 16
    assert plan["fill_tokens"] == [computed]


def test_rows_ask_forty_steps_and_the_tokenizer_cache_holds_every_prompt(tmp_path):
    # Three tokens a step and four, alternately: 3.5 a step.
    record = _record(tmp_path, [[1000] * 20], advances=(3, 4), body=3)
    plan = design(record, 1)
    assert plan["max_tokens"] == 140
    # Each distinct prompt at 4 bytes a character, 3 characters a token, and
    # 12 bytes an id.
    sent = {(i, t) for g in plan["groups"] for i, t in g["rows"][0]}
    held = sum(t for _, t in sent) * (4 * 3 + 12)
    assert plan["tokenize_cache_mb"] == math.ceil(held / 2**20)


def test_the_same_record_and_seed_print_the_same_bytes(tmp_path, capsys):
    record = _record(tmp_path, [_spread(1000, 90_000)], histories=[_spread(16, 900)])
    printed = []
    for kind, seed in ("design", "4"), ("design", "4"), ("design", "5"), ("trace", "4"):
        main([kind, str(record), seed])
        printed.append(capsys.readouterr().out)
    assert printed[0] == printed[1] != printed[2]
    assert json.loads(printed[0]) == design(record, 4)
    assert json.loads(printed[3]) == trace_design(record, 4)
    with pytest.raises(SystemExit):
        main(["design", str(record)])


@pytest.mark.parametrize("model, replay", [("qwen3", 0), ("deepseek_v41", BLOCK)])
def test_trace_points_run_cold_then_each_list_bottom_first_within_the_budget(
    tmp_path, model, replay
):
    # A budget of 64 tokens and five sequences: up to four rows beside a parked
    # decode; buckets at 16 and 64.
    engine = _engine(max_num_seqs=5, prefill_buckets=[16, 64])
    histories = [_spread(16, 4000)]
    record = _record(tmp_path, [[100] * 4], histories, engine, model=model)
    plan = trace_design(record, 1)
    assert plan["hit_replay"] == replay
    levels = list(np.percentile(sorted(histories[0]), (10, 50, 90)))
    assert plan["levels"] == levels
    points = plan["points"]
    cold = [p for p in points if p["level"] is None]
    assert points[: len(cold)] == cold
    shapes = [(len(p["rows"]), p["rows"][0][2]) for p in cold]
    assert shapes == sorted(shapes)
    assert (1, 17) in shapes and (1, 65) not in shapes
    assert {n for n, _ in shapes} == {1, 2, 4}
    assert all(n * q <= 64 for n, q in shapes)
    assert all(p["record"] == [[q, q]] * n for p, (n, q) in zip(cold, shapes))
    warm = points[len(cold) :]
    # No scaling on this pool: one list, bottom level first, n and q ascending.
    order = [(p["level"], len(p["rows"]), p["rows"][0][2]) for p in warm]
    assert order == sorted(order)
    for p in warm:
        n, q = len(p["rows"]), p["rows"][0][2]
        assert [r[0] for r in p["rows"]] == list(range(n))
        heights = [plan["sessions"][i]["cuts"][p["level"]] for i in range(n)]
        assert [r[1] for r in p["rows"]] == heights
        assert all(h % BLOCK == 0 and h > 0 for h in heights)
        # A row computes its q tokens, and the block below its history where
        # a hit replays one, as its step records its first chunk: a row past
        # the budget computes the budget first.
        first = min(q + replay, 64)
        assert p["record"] == [[first, h - replay + first] for h in heights]
        assert n * (q + replay) <= 64 or (n, q + replay) == (1, 65)
    shapes = {(len(p["rows"]), p["rows"][0][2]) for p in warm}
    assert {(1, 64 - replay), (1, 65 - replay)} <= shapes
    # Each session fills to its top history once, from the level below, and
    # each later fill computes its history's next token alone, with the block
    # below where a hit replays one.
    fills = sum(len(p["rows"]) for p in warm)
    used = {r[0] for p in warm for r in p["rows"]}
    tops = sum(plan["sessions"][i]["cuts"][2] + 1 for i in used)
    assert plan["fill_tokens"] == tops + (fills - len(used)) * (1 + replay)


@pytest.mark.parametrize("ranks, stage", [(2, False), (1, True)])
def test_a_trace_design_refuses_a_dp_or_pipeline_record(tmp_path, ranks, stage):
    record = _record(tmp_path, [[100]] * ranks, histories=[[64]] * ranks)
    if stage:
        (record / "steps" / "pp1.dp0.jsonl").write_text("")
    with pytest.raises(ValueError, match="DP 1 and PP 1"):
        trace_design(record, 1)
