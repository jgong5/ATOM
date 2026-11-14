# SPDX-License-Identifier: MIT
"""`compass advice SPACE OUT`, through the `compass` entry: a space file
expanded into its candidates and points, and each way a space is refused."""

import copy
import json

import pytest

from atom.compass import cli

MODEL = "/models/deepseek-ai/DeepSeek-V4-Pro"
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SINGLE = "recipes/DeepSeek-V4-Agentic-InferenceX.md"
MESH = "recipes/mesh/DeepSeek-V4.md"
SEQS = "a fixed 512 keeps one readings entry per configuration; the recipe has 2x"
PREFIX = "on, as the agentic workload and the single-node recipe have it"
UTIL = "0.75 per decode side, Mooncake's dmabuf limit on the development nodes"
BLOCK = "not set: ATOM forces 256 on any DeepSeek-V4 checkpoint"
MTP3 = {
    "--method": "mtp",
    "--num-speculative-tokens": 3,
    "--spec-decode-acceptance-length": 2.49,
}
ONE_NODE = {
    "--index_cache_dtype": "fp4",
    "--gpu-memory-utilization": 0.9,
    "--max-num-batched-tokens": 16384,
    "--attn-prefill-chunk-size": 16384,
    "--level": 3,
    "--cudagraph-mode": "FULL",
    **MTP3,
}
DPA = {**ONE_NODE, "--enable-dp-attention": True, "--enable-tbo": True}
TBO_ENV = {"GPU_MAX_HW_QUEUES": 5, "ATOM_NUMA_BIND": 1}
DP_ENV = {
    **TBO_ENV,
    "ATOM_DP_SESSION_AFFINITY": 1,
    "ATOM_DP_LB_REQ_EQUIV": 512,
    "ATOM_ENABLE_PREFILL_DELAYER": 1,
    "ATOM_PREFILL_DECODE_INTERVAL": 10,
}
MESH_DEPARTS = {"--enable-prefix-caching": PREFIX, "--block-size": BLOCK}
#: The DeepSeek-V4-Pro development template's five configurations, one value
#: each of one dimension, with their recipes' main arguments.
TEMPLATE = {
    "model": MODEL,
    "machine": {"device": "mi355x", "ptl": "disabled"},
    "store": "/data/compass/store",
    "workload": {
        "traces": "/data/cc-traces/traces.jsonl",
        "select": "0-255",
        "concurrency": CONCURRENCY,
    },
    "slo": {"min_interactivity_p90": 30, "max_ttft_p90_s": 5},
    "common": {
        "args": {
            "--tensor-parallel-size": 8,
            "--kv_cache_dtype": "fp8",
            "--enable_prefix_caching": True,
            "--max-num-seqs": 512,
        },
        "env": {"AITER_BF16_FP8_MOE_BOUND": 0, "ATOM_MOE_GU_ITLV": 1},
    },
    "dimensions": {
        "config": {
            "S1": {
                "basis": f"{SINGLE}, TP",
                "args": ONE_NODE,
                "departures": {"--max-num-seqs": SEQS},
            },
            "S2": {
                "basis": f"{SINGLE}, DP attention",
                "args": DPA,
                "env": DP_ENV,
                "departures": {"--max-num-seqs": SEQS},
            },
            "S5": {
                "basis": f"{SINGLE}, DP attention with LMCache",
                "args": DPA,
                "env": {**DP_ENV, "ATOM_KV_OFFLOAD": "lmcache_mp"},
                "departures": {"--max-num-seqs": SEQS},
            },
            "P1": {
                "basis": f"{MESH}, pure TP",
                "deployment": "1p1d",
                "args": {"--trust-remote-code": True},
                "prefill": {"args": {"--gpu-memory-utilization": 0.75}},
                "decode": {"args": {"--gpu-memory-utilization": 0.75}},
                "departures": {**MESH_DEPARTS, "--gpu-memory-utilization": UTIL},
            },
            "P3": {
                "basis": f"{MESH}, DPA",
                "deployment": "1p1d",
                "args": {"--trust-remote-code": True, "--enable-dp-attention": True},
                "prefill": {
                    "args": {"--gpu-memory-utilization": 0.72, "--enable-tbo": True},
                    "env": TBO_ENV,
                },
                "decode": {"args": {"--gpu-memory-utilization": 0.75}},
                "departures": {
                    **MESH_DEPARTS,
                    "--gpu-memory-utilization": UTIL + "; 0.72 on prefill",
                },
            },
        }
    },
}


def advise(tmp_path, space, out="out") -> dict:
    path = tmp_path / "space.json"
    path.write_text(space if isinstance(space, str) else json.dumps(space))
    cli.main(["advice", str(path), str(tmp_path / out)])
    return json.loads((tmp_path / out / "candidates.json").read_text())


def refusal(tmp_path, space) -> str:
    with pytest.raises(SystemExit) as refused:
        advise(tmp_path, space)
    assert not (tmp_path / "out").exists()
    return str(refused.value)


def small(**fields) -> dict:
    """A space of two candidates, with `fields` in place of its own."""
    space = {
        "model": "/m",
        "machine": {"device": "mi355x", "ptl": "disabled"},
        "store": "/s",
        "workload": {"traces": "/t.jsonl", "select": "agentx", "concurrency": [1, 4]},
        "slo": {},
        "common": {"basis": "a recipe", "args": {"--max-num-seqs": 512}},
        "dimensions": {
            "tp": {
                "tp4": {"args": {"--tensor-parallel-size": 4}},
                "tp8": {"args": {"--tensor-parallel-size": 8}},
            }
        },
    }
    return copy.deepcopy({**space, **fields})


def test_the_template_expands_twice_into_the_same_candidates_and_points(
    tmp_path, capsys
):
    first = advise(tmp_path, TEMPLATE, "first")
    second = advise(tmp_path, TEMPLATE, "second")

    written = [
        (tmp_path / d / "candidates.json").read_bytes() for d in ("first", "second")
    ]
    assert written[0] == written[1] and first == second
    names = ["S1", "S2", "S5", "P1", "P3"]
    assert [c["name"] for c in first["candidates"]] == names
    assert [p["name"] for p in first["points"]] == [
        f"{c}-c{n}" for c in names for n in CONCURRENCY
    ]
    s1, _, s5, p1, p3 = first["candidates"]
    assert (s1["deployment"], s1["basis"]) == ("single", f"{SINGLE}, TP")
    assert s1["departures"] == {"--max-num-seqs": SEQS}
    assert s1["instances"]["server"]["args"][:6] == [
        "--model",
        MODEL,
        "--tensor-parallel-size",
        "8",
        "--kv-cache-dtype",
        "fp8",
    ]
    assert s5["instances"]["server"]["env"]["ATOM_KV_OFFLOAD"] == "lmcache_mp"
    assert list(p1["instances"]) == ["prefill", "decode"]
    prefill, decode = p3["instances"]["prefill"], p3["instances"]["decode"]
    assert "--enable-tbo" in prefill["args"] and "--enable-tbo" not in decode["args"]
    assert (
        prefill["env"]["GPU_MAX_HW_QUEUES"] == "5"
        and "GPU_MAX_HW_QUEUES" not in decode["env"]
    )
    assert decode["args"][-2:] == ["--gpu-memory-utilization", "0.75"]
    assert first["points"][-1]["instances"] == {
        side: {"args": one["args"], "env": one["env"]}
        for side, one in p3["instances"].items()
    }
    out = capsys.readouterr().out
    assert (
        f"P3: 1p1d, {MESH}, DPA\n  --enable-prefix-caching departs: {PREFIX}\n" in out
    )
    assert out.endswith(
        f"5 candidates and 45 points in {tmp_path / 'second' / 'candidates.json'}\n"
    )


def test_a_product_less_its_exclusions_keeps_the_dimensions_order(tmp_path):
    space = small(
        dimensions={
            "tp": {"tp4": {}, "tp8": {}},
            "mtp": {"off": {}, "mtp1": {}, "mtp3": {}},
        },
        exclude=[{"tp": "tp4", "mtp": "mtp1"}, {"mtp": "mtp3", "tp": "tp8"}],
    )

    expansion = advise(tmp_path, space)

    assert [c["name"] for c in expansion["candidates"]] == [
        "tp4-off",
        "tp4-mtp3",
        "tp8-off",
        "tp8-mtp1",
    ]
    assert expansion["candidates"][1]["values"] == {"tp": "tp4", "mtp": "mtp3"}


def test_a_per_concurrency_argument_is_its_factor_times_each_point_s_concurrency(
    tmp_path,
):
    space = small(
        common={"basis": "a recipe", "per_concurrency": {"--max-num-seqs": 2}}
    )

    expansion = advise(tmp_path, space)

    server = expansion["candidates"][0]["instances"]["server"]
    assert server["per_concurrency"] == {"--max-num-seqs": 2}
    assert "--max-num-seqs" not in server["args"]
    assert [p["instances"]["server"]["args"][-2:] for p in expansion["points"]] == [
        ["--max-num-seqs", "2"],
        ["--max-num-seqs", "8"],
    ] * 2


@pytest.mark.parametrize(
    "change, named",
    [
        (lambda s: s.update(searched=1), "space.json: unknown field 'searched'"),
        (lambda s: s["machine"].update(gpus=8), "machine: unknown field 'gpus'"),
        (lambda s: s["workload"].update(trace=1), "workload: unknown field 'trace'"),
        (lambda s: s["slo"].update(ttft=1), "slo: unknown field 'ttft'"),
        (lambda s: s["common"].update(arg={}), "common: unknown field 'arg'"),
        (
            lambda s: s["dimensions"]["tp"]["tp8"].update(prefill={"basis": "b"}),
            "dimensions.tp.tp8.prefill: unknown field 'basis'",
        ),
        (lambda s: s.pop("slo"), "space.json: no 'slo'"),
        (
            lambda s: s["common"]["args"].update({"-tp": 8}),
            "common.args: '-tp' takes a long flag",
        ),
        (
            lambda s: s["workload"].update(select="9-3"),
            "workload.select takes A-B, A at most B, or agentx",
        ),
    ],
)
def test_an_unknown_or_malformed_field_is_refused_by_name(tmp_path, change, named):
    space = small()
    change(space)

    assert named in refusal(tmp_path, space)


@pytest.mark.parametrize(
    "common, tp8, named",
    [
        (
            {},
            {"args": {"--max_num_seqs": 256}},
            "tp8: --max-num-seqs 512 in common, but --max-num-seqs 256 in dimensions",
        ),
        (
            {},
            {"args": {"--model": "/other"}},
            "candidate tp8: --model /m in model, but --model /other in dimensions.tp.tp8",
        ),
        (
            {},
            {"per_concurrency": {"--max-num-seqs": 2}},
            "512 in common, but --max-num-seqs 2x the concurrency in dimensions.tp.tp8",
        ),
        (
            {},
            {
                "args": {
                    "--no-enable_prefix_caching": True,
                    "--enable-prefix-caching": True,
                }
            },
            "--no-enable-prefix-caching in dimensions.tp.tp8, but --enable-prefix-",
        ),
        (
            {"deployment": "single"},
            {"deployment": "1p1d"},
            "deployment single in common, but deployment 1p1d in dimensions.tp.tp8",
        ),
        (
            {},
            {"basis": "another recipe"},
            "basis a recipe in common, but basis another recipe in dimensions.tp.tp8",
        ),
        (
            {"departures": {"--level": "a"}},
            {"departures": {"--level": "b"}},
            "--level departs: a in common, but --level departs: b in dimensions.tp.tp8",
        ),
        (
            {},
            {"deployment": "1p1d", "env": {"A": 1}, "decode": {"env": {"A": 2}}},
            "A=1 in dimensions.tp.tp8, but A=2 in dimensions.tp.tp8.decode",
        ),
        (
            {},
            {"prefill": {}},
            "candidate tp8 is single-node, and dimensions.tp.tp8 gives it a prefill side",
        ),
    ],
)
def test_contradictory_values_are_refused_naming_both(tmp_path, common, tp8, named):
    space = small()
    space["common"].update(common)
    space["dimensions"]["tp"]["tp8"].update(tp8)

    assert named in refusal(tmp_path, space)


def test_one_key_twice_or_one_concurrency_twice_is_refused(tmp_path):
    text = json.dumps(small()).replace('"store": "/s"', '"store": "/s", "store": "/r"')
    assert "'store' is given twice in one object" in refusal(tmp_path, text)

    space = small(workload={"traces": "/t", "select": "agentx", "concurrency": [4, 4]})
    assert "workload.concurrency lists 4 twice" in refusal(tmp_path, space)


def test_an_excluded_combination_is_not_judged(tmp_path):
    space = small(exclude=[{"tp": "tp8"}])
    space["dimensions"]["tp"]["tp8"]["args"]["--max-num-seqs"] = 256

    expansion = advise(tmp_path, space)

    assert [c["name"] for c in expansion["candidates"]] == ["tp4"]


@pytest.mark.parametrize(
    "fields, named",
    [
        ({"dimensions": {}}, "dimensions names none, so it expands to no candidate"),
        ({"dimensions": {"tp": {}}}, "dimensions.tp has no value"),
        (
            {"exclude": [{"tp": "tp4"}, {"tp": "tp8"}]},
            "the exclusions remove every combination of the dimensions",
        ),
        (
            {"workload": {"traces": "/t", "select": "0-9", "concurrency": []}},
            "workload.concurrency lists none, so it expands to no point",
        ),
        ({"exclude": [{"pp": "pp2"}]}, "exclude[0]: unknown dimension 'pp'"),
        ({"exclude": [{"tp": "tp2"}]}, "exclude[0]: tp has no value 'tp2'"),
        (
            {"common": {}},
            "candidate tp4 names no ATOM basis",
        ),
    ],
)
def test_an_empty_expansion_or_a_candidate_with_no_basis_is_refused(
    tmp_path, fields, named
):
    assert named in refusal(tmp_path, small(**fields))


def test_compass_advice_reads_only_the_space_it_is_given(tmp_path):
    with pytest.raises(SystemExit, match="^compass advice: .*No such file") as refused:
        cli.main(["advice", str(tmp_path / "absent.json"), str(tmp_path / "out")])

    assert str(tmp_path / "absent.json") in str(refused.value)
    assert not (tmp_path / "out").exists()
