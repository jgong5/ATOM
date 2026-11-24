# SPDX-License-Identifier: MIT
"""The unified calibration procedure v1.1's design tooling
(wuhuikx/LLM-Booster#739): `atom.compass.calib`'s config and the deployment it
reads from a record (`deploy`), and each step's column counts (`columns`), on
synthetic configs, records and laws."""

import copy
import json
from pathlib import Path

import pytest

from atom.compass.backends.shape import DUTY_TAU, BatchView, RequestShape
from atom.compass.backends.step_grid_plan import TARGET
from atom.compass.backends.step_linear import StepLinearBackend, StepLinearLaw
from atom.compass.calib import columns, deploy

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
                    {"pin": 0, "groups": [{"rows": {"seqs_share": 1.0}, "prompt": 512}]}
                ],
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


def _dep(**over):
    raw = _config()
    raw["deployment"] = {**DEPLOYMENT, **over}
    return deploy.parse(raw).deployment


# -- the config ---------------------------------------------------------------


def test_a_config_states_its_intercept_first():
    for kind in ("decode", "prefill"):
        raw = copy.deepcopy(_config())
        raw["columns"][kind] = raw["columns"][kind][1:] + ["step"]
        with pytest.raises(ValueError, match="intercept"):
            deploy.parse(raw)


def test_a_config_names_each_segment_once():
    raw = _config()
    raw["segments"] = raw["segments"] + [dict(raw["segments"][0])]
    with pytest.raises(ValueError, match="repeat"):
        deploy.parse(raw)


def test_a_config_tightens_an_acceptance_bound_and_keeps_the_others():
    cfg = deploy.parse(_config(acceptance={"vif": 5, "g": {"decode": 0.005}}))
    assert cfg.acceptance["g"] == {"decode": 0.005, "prefill": 0.02}
    assert cfg.acceptance["vif"] == 5
    assert cfg.acceptance["rung_steps"] == 60 and cfg.acceptance["levels"] == 3
    assert cfg.range["min_context"] == 256 and cfg.range["min_seq_tokens"] == 64
    loose = deploy.parse(_config(acceptance={}, range={}, timing={}))
    assert loose.acceptance["g"] == {"decode": 0.01, "prefill": 0.02}
    assert loose.range == deploy.RANGE and loose.timing == deploy.TIMING


def test_a_config_extends_another_and_inserts_segments_into_it(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps(_config()))
    mid = {
        "name": "mid",
        "extends": "base.json",
        "noise": {"decode": 0.01, "prefill": 0.01},
        "insert": [
            {"after": "D-a", "segments": [{"name": "A-1", "kind": "copy", "of": "D-a"}]}
        ],
    }
    top = {
        "name": "top",
        "extends": "mid.json",
        "insert": [
            {"after": None, "segments": [{"name": "A-0", "kind": "copy", "of": "D-a"}]},
            {
                "after": "P-r",
                "segments": [{"name": "A-2", "kind": "copy", "of": "D-b"}],
            },
        ],
    }
    (tmp_path / "mid.json").write_text(json.dumps(mid))
    (tmp_path / "top.json").write_text(json.dumps(top))
    cfg = deploy.load(tmp_path / "top.json")
    assert cfg.name == "top" and cfg.deployment.dp == 2
    assert cfg.noise == {"decode": 0.01, "prefill": 0.01}  # mid's, over base's
    assert [s["name"] for s in cfg.segments] == [
        "A-0",
        "D-a",
        "A-1",
        "D-b",
        "P-r",
        "A-2",
    ]
    top["insert"] = [{"after": "no-such", "segments": []}]
    (tmp_path / "top.json").write_text(json.dumps(top))
    with pytest.raises(ValueError):
        deploy.load(tmp_path / "top.json")


# -- the deployment -----------------------------------------------------------


def test_a_deployment_names_the_rung_a_step_replays_and_its_cache_unit():
    dep = _dep(rungs=[1, 2, 4, 8], max_num_seqs=10)
    assert [dep.rung(n) for n in (1, 3, 4, 8, 9)] == [1, 4, 4, 8, None]
    assert list(dep.rows_of(1)) == [1] and list(dep.rows_of(8)) == [5, 6, 7, 8]
    assert dep.prefix_unit == 128 and dep.cached(300) == 256  # whole blocks
    assert dep.query == 2 and dep.per_step == 1.5
    assert _dep(drafts=0).per_step == 1.0
    assert _dep(prefix_unit=4096).cached(9000) == 8192
    with pytest.raises(ValueError, match="ascend"):
        _dep(rungs=[1, 4, 2])
    with pytest.raises(ValueError, match="below one token"):
        _dep(acceptance=0.9)


def test_the_kv_pool_is_a_number_or_a_run_json_engines(tmp_path, monkeypatch):
    run = {"engine": {"num_kvcache_blocks": 1000, "kv_cache_block_size": 64}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv("CALIB_RUN", str(tmp_path))
    assert deploy.kv_tokens(5000, 128) == 5000
    assert deploy.kv_tokens({"run_json": "$CALIB_RUN/run.json"}, 128) == 64000
    raw = _config()
    raw["deployment"] = {**DEPLOYMENT, "kv_tokens": {"run_json": "$CALIB_RUN/run.json"}}
    cfg = deploy.parse(raw)
    assert cfg.deployment.kv_tokens == 64000
    assert cfg.raw["deployment"]["kv_tokens"] == 64000  # stated plain


def test_a_count_is_a_share_of_max_num_seqs_a_rank_or_the_group():
    dep = _dep(max_num_seqs=64, dp=4)
    assert deploy.count(7, dep) == 7
    assert deploy.count({"seqs_share": 1.0}, dep) == 64
    assert deploy.count({"seqs_share": 1.0, "minus": 9}, dep) == 55
    assert deploy.count({"seqs_share": 0.125}, dep) == 8
    assert deploy.count({"seqs_share": 1.0, "dp": True}, dep) == 256
    assert deploy.count({"seqs_share": 1.0, "per_rank": -1, "dp": True}, dep) == 252
    assert deploy.count({"per_rank": 1, "dp": True}, dep) == 4
    assert deploy.count({"seqs_share": 0.5, "plus": 2}, dep) == 34
    with pytest.raises(ValueError, match="takes"):
        deploy.count({"seqs": 1}, dep)
    with pytest.raises(ValueError, match="is 0"):
        deploy.count({"seqs_share": 1.0, "minus": 64}, dep)


def test_a_token_size_is_a_share_of_the_pool_the_model_the_budget_or_a_bucket():
    dep = _dep(kv_tokens=1_000_000, max_model_len=65536, max_num_batched_tokens=4096)
    assert deploy.tokens(1000, dep) == 1000  # a number is taken as given
    assert deploy.tokens({"kv_share": 0.5}, dep, over=10) == deploy.whole(50_000, dep)
    # held under LEN_CAP of the model length
    assert deploy.tokens({"kv_share": 0.5}, dep) == deploy.whole(0.9 * 65536, dep)
    assert deploy.tokens({"kv_share": 0.5, "over": 20}, dep) == 25024
    assert deploy.tokens({"len_share": 0.5}, dep) == 32768
    # under the pool split over the rows
    assert deploy.tokens({"len_share": 1.0}, dep, over=100) == deploy.whole(10_000, dep)
    assert deploy.tokens({"budget_share": 0.75}, dep) == 3072
    assert deploy.tokens({"bucket_share": 1.0}, dep) == 1024
    assert deploy.whole(10_000, dep) % 64 == 0 and deploy.whole(1, dep) == 64
    with pytest.raises(ValueError, match="buckets"):
        deploy.tokens({"bucket_share": 1.0}, _dep(prefill_buckets=[]))
    with pytest.raises(ValueError, match="takes"):
        deploy.tokens({"kv": 0.5}, dep)
    with pytest.raises(ValueError, match="no share"):
        deploy.tokens({"over": 2}, dep)


# -- a deployment read from a record ----------------------------------------------


def _record_dir(path: Path, kda: bool = False, forced: bool = True, cal: bool = False):
    """A record as `compass record` lays it out (with `cal`, under the calibration
    run's real/), its model's config.json beside it: each rank's engine file,
    and run.json's engine, the `TARGET` fields of rank 0's at the fewest KV
    blocks any rank states, as `drive.record.write_engine_and_costs` writes it."""
    model = path / "model"
    model.mkdir(parents=True)
    hf = {"n_routed_experts": 64, "num_experts_per_tok": 4, "top_k": 50}
    if kda:
        hf = {"text_config": {"num_experts": 32, "num_experts_per_token": 8}}
        hf["text_config"]["linear_attn_config"] = {"kda_layers": [1, 2]}
    (model / "config.json").write_text(json.dumps(hf))
    rec = path / "rec" / ("real" if cal else "")
    (rec / "steps").mkdir(parents=True)
    serve = [
        "--model",
        str(model),
        "-tp",
        "4",
        "--enable-dp-attention",
        "--attn-prefill-chunk-size",
        "8192",
        "--state-checkpoint-interval-tokens",
        "4096",
        "--decode-context-parallel-size",
        "2",
        "--num-speculative-tokens",
        "3",
    ]
    if forced:
        serve += ["--spec-decode-acceptance-length", "2.5"]
    engines = {}
    for rank, blocks in ((0, 1001), (1, 1000)):
        engines[rank] = {
            "speculative": ["dspark", 3],
            "ladder": [1, 2, 4, 8, 16, 32],
            "data_parallel_size": 4,
            "tensor_parallel_size": 1,
            "max_num_seqs": 32,
            "max_num_batched_tokens": 8192,
            "max_model_len": 131072,
            "kv_cache_block_size": 64,
            "num_kvcache_blocks": blocks,
            "prefill_buckets": [256, 512],
        }
        (rec / "steps" / f"dp{rank}.engine.json").write_text(json.dumps(engines[rank]))
    engine = {k: engines[0][k] for k in TARGET}
    engine["num_kvcache_blocks"] = min(
        e["num_kvcache_blocks"] for e in engines.values()
    )
    run = {"argv": ["-c", "record", "--gpu", "0", "--", *serve], "engine": engine}
    (rec / "run.json").write_text(json.dumps(run))
    (rec / "steps" / "dp0.acceptance.json").write_text(
        json.dumps({"draft_tokens": 3, "distribution": {"1": 1, "2": 3}})
    )
    return path / "rec"


def test_a_record_states_its_deployment(tmp_path):
    got = deploy.from_record(_record_dir(tmp_path))
    assert got == {
        "max_num_seqs": 32,
        "max_num_batched_tokens": 8192,
        "max_model_len": 131072,
        "block_size": 64,
        "dp": 4,
        "tp": 1,
        "rungs": [1, 2, 4, 8, 16, 32],
        "kv_tokens": 1000 * 64,  # run.json's engine, over its rank 0's
        "prefill_buckets": [256, 512],
        "chunk": 8192,
        "dcp": 2,
        "drafts": 3,
        "acceptance": 2.5,
        "moe_routing": [64, 4],
    }


def test_a_kda_model_hits_at_its_state_checkpoint_and_acceptance_falls_back(tmp_path):
    got = deploy.from_record(_record_dir(tmp_path, kda=True, forced=False, cal=True))
    assert got["moe_routing"] == [32, 8] and got["prefix_unit"] == 4096
    assert got["acceptance"] == pytest.approx(1 + (1 * 1 + 2 * 3) / 4)


def test_a_field_beside_from_record_overrides_it_and_a_given_record_replaces_the_config(
    tmp_path,
):
    rec = _record_dir(tmp_path)
    raw = _config()
    raw["deployment"] = {
        "from_record": str(rec),
        "max_num_seqs": 16,
        "trace_model": "m",
    }
    cfg = deploy.parse(raw)
    dep = cfg.deployment
    assert dep.max_num_seqs == 16 and dep.trace_model == "m"
    assert dep.rungs == (1, 2, 4, 8, 16, 32) and dep.dcp == 2
    assert cfg.raw["deployment"]["kv_tokens"] == 64000  # stated plain
    raw["deployment"] = dict(DEPLOYMENT)
    followed = deploy.parse(raw, record=rec).deployment
    assert followed.max_num_seqs == 32 and followed.dp == 4 and followed.dcp == 2
    assert followed.trace_block == 64  # what a record does not state stays the config's
    (tmp_path / "cfg.json").write_text(json.dumps(raw))
    assert deploy.load(tmp_path / "cfg.json", record=rec).deployment == followed


# -- the columns --------------------------------------------------------------


def _vector(coefficients: dict, **extra) -> dict:
    axes = [f for f in coefficients if f != "step"]
    return {
        "coefficients": coefficients,
        "hull": {"equal": dict.fromkeys(axes, 0.0), "facets": []},
        **extra,
    }


DECODE_LAW = {
    "step": 0.004,
    "context": 2e-9,
    "graph_padding": 5e-10,
    "moe_experts": 3e-6,
}
PREFILL_LAW = {
    "step": 0.01,
    "tokens": 6e-6,
    "query_square": 2e-11,
    "query_cached": 1e-11,
    "history": 5e-9,
    "moe_rows": 2e-6,
    "cold_peer_rows": 1e-6,
    "seqs": 3e-4,
    "out": 2e-3,
    "padding": 4e-7,
}


def test_a_column_the_backend_prices_takes_the_backends_value_on_the_same_step():
    """Each column's count times a fitted law's coefficient is the term the
    backend prices that law at, on the same step."""
    law = {
        "provenance": "synthetic",
        "prefill": _vector(PREFILL_LAW, duty_tau=DUTY_TAU, buckets=[256, 512, 1024]),
        "decode": {"4": _vector(DECODE_LAW)},
        "moe_routing": {"experts": 16, "top_k": 2},
    }
    backend = StepLinearBackend(StepLinearLaw(law), dp=2)
    dep = _dep()
    decode = BatchView(
        tuple(RequestShape(2, c, True) for c in (1000, 2000, 3000)),
        capture_rung=4,
        moe_rows=16,
        moe_tokens=10,
        produces_output=True,
        dp_rank=0,
    )
    prefill = BatchView(
        (RequestShape(300, 1300, False), RequestShape(100, 100, False)),
        moe_rows=900,
        group_max_tokens=450,
        prefill_duty=0.25,
        produces_output=True,
        dp_rank=0,
    )
    for kind, view, coefficients in (
        ("decode", decode, DECODE_LAW),
        ("prefill", prefill, PREFILL_LAW),
    ):
        names = list(coefficients)
        got = dict(zip(names, columns.values(view, kind, names, dep)))
        priced = {t.name: t.seconds for t in backend.price(view).terms}
        assert set(priced) == {f"{kind}.{n}" for n in names}
        for n in names:
            assert priced[f"{kind}.{n}"] == pytest.approx(coefficients[n] * got[n]), n
        assert columns.provisional(kind, names) == []
    # the counts are the step's own, not the law's ones
    assert columns.values(decode, "decode", ["step", "context"], dep) == (1.0, 6000.0)
    assert columns.values(prefill, "prefill", ["padding", "history"], dep) == (
        512 - 400,
        1000.0,
    )


def test_an_opt_in_column_the_backend_lacks_is_counted_by_its_provisional_definition():
    dep = _dep()
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
        ["step", "context", "graph_padding", "row_padding", "peer_tokens", "rows"],
        dep,
    )
    assert got == (1.0, 4000.0, view.graph_padding, 2 * 3000 - 4000, 10 - 4, 2.0)
    assert columns.provisional("decode", ["step", "context", "no_such"]) == ["no_such"]
    with pytest.raises(ValueError, match="neither"):
        columns.values(view, "decode", ["step", "no_such_column"], dep)
    gather = ["step", "gather_rows", "moe_imbalance"]
    eager = BatchView(
        (RequestShape(2000, 2000, False),),
        moe_rows=3000,
        group_max_tokens=2000,  # past the largest bucket: the variable-length gather
        produces_output=True,
        dp_rank=0,
    )
    assert columns.values(eager, "prefill", gather, dep, 0.2) == (1.0, 3000.0, 1000.0)
    assert columns.values(eager, "prefill", gather, dep, 0.5) == (1.0, 3000.0, 0.0)
    graphed = BatchView(
        (RequestShape(450, 450, False),),
        moe_rows=800,
        group_max_tokens=450,
        produces_output=True,
        dp_rank=0,
    )
    assert columns.values(graphed, "prefill", gather, dep) == (1.0, 2.0 * 512, 0.0)


def test_a_step_the_backend_refuses_is_refused_by_name():
    no_gather = BatchView(
        (RequestShape(300, 300, False),),
        group_max_tokens=300,
        moe_tokens=300,
        produces_output=True,
        dp_rank=0,
    )
    names = ["step", "tokens", "query_square", "query_cached", "moe_rows"]
    assert columns.values(no_gather, "prefill", names[:4], _dep()) == (
        1.0,
        300.0,
        300.0**2,
        0.0,
    )
    with pytest.raises(ValueError, match="refuses the prefill view"):
        columns.values(no_gather, "prefill", names, _dep())
