"""q16 source controls add only their qualified queries and native histories."""

import hashlib
import json

import pytest

from atom.compass.core.cost.cached_q16 import CachedQ16Prices, GDN, MHA, GATHER
from atom.compass.core.cost.families import ParametricPriceLibrary
from atom.compass.core.cost.library import PriceLibrary
from atom.compass.runtime.microbench import signature_of


def gdn(layer=0, query=16, read=0, write=1):
    return {"name": GDN, "input_shapes": [[query, 10240], [query, 48]],
            "dtypes": ["bfloat16", "bfloat16"], "layouts": [],
            "scalars": [["#4", f"language_model.model.layers.{layer}.linear_attn"]],
            "context": [["num_prefills", 1], ["num_prefill_tokens", query],
                        ["num_decodes", 0], ["num_decode_tokens", 0],
                        ["num_actual_tokens", query], ["replayssm", False],
                        ["has_initial_state", [[1], "bool"]],
                        ["non_spec_query_start_loc", [[0, query], "int32"]],
                        ["non_spec_state_indices_tensor", [[write], "int32"]],
                        ["non_spec_state_indices_in_tensor", [[read], "int32"]]]}


def mha(history=65536, layer=3, query=16):
    return {"name": MHA, "input_shapes": [[query, 6144], [query, 1024]],
            "dtypes": ["bfloat16", "bfloat16"], "layouts": [],
            "output_shapes": [[query, 6144]], "output_dtypes": ["bfloat16"],
            "scalars": [["#5", f"language_model.model.layers.{layer}.self_attn"]],
            "context": [["context_lens", [history]],
                        ["cu_seqlens_q", [0, query]], ["cu_seqlens_k", [0, history]],
                        ["max_seqlen_q", query], ["max_seqlen_k", history],
                        ["total_kv", history], ["num_cached_tokens", [history - query]],
                        ["has_cached", True], ["is_prefill", True],
                        ["state", "prefill_prefix"], ["block_tables_shape", [1, 16384]],
                        ["slot_mapping", list(range(history - query, history))]]}


def gather(index=15):
    return {"name": GATHER, "input_shapes": [[16, 5120], [1]],
            "dtypes": ["bfloat16", "int32"], "layouts": [],
            "scalars": [], "context": [], "int_values": [[1, [index]]],
            "output_shapes": [[1, 5120]], "output_dtypes": ["bfloat16"]}


@pytest.fixture
def bundle(tmp_path):
    entries = []
    cases = [("gdn_fork_ref", "gdn", [gdn(layer=l) for l in range(64) if l % 4 != 3]),
             ("gather_h16_m1", "gather", [gather()])]
    cases += [(f"mha_h{h}", "mha", [mha(h, l) for l in range(3, 64, 4)])
              for h in (32768, 65536, 131072)]
    for case, family, ops in cases:
        timer, regions = ("over", 8) if family == "mha" else ("graph", 1)
        graph = {"ops": ops, "key": {"topology": [["tp", 1]]}}
        prices = {"provenance": {"topology": {"tp": 1}}, "unpriced": {},
                  "prices": {signature_of(op): {"name": op["name"],
                             "seconds": (i + 1) * 1e-5 + (
                                 dict(op["context"])["total_kv"] * 1e-9 if family == "mha" else 0),
                             "kernels": {}, "cache": timer, "kv_regions": regions}
                             for i, op in enumerate(ops)}}
        entry = {"case_id": case, "family": family, "source_qualified": True,
                 "observed_cache": timer, "kv_regions": regions}
        for key, value in (("graph", graph), ("price", prices)):
            path = tmp_path / f"{case}.{key}.json"
            path.write_text(json.dumps(value))
            entry[key] = {"path": "/original/source/" + path.name,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        entries.append(entry)
    handoff = {"schema": "compass.q16_reference_export/1", "source_qualified": True,
               "all_error_gates_pass": True, "fit_inputs_are_references_only": True,
               "heldout_timings_used_as_fit_inputs": False,
               "scope": {"model": "Qwen/Qwen3.8-27B", "topology": {"tp": 1},
                         "query_tokens": 16, "num_sequences": 1, "dtype": "bfloat16"},
               "entries": entries}
    path = tmp_path / "HANDOFF.json"
    path.write_text(json.dumps(handoff))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_interpolation_preserves_layer_and_source_anchors(bundle):
    library = CachedQ16Prices(PriceLibrary(), *bundle)
    for h in (32768, 49152, 65536, 98304, 131072):
        for i, layer in enumerate(range(3, 64, 4)):
            record, _ = library.lookup(mha(h, layer), {"tp": 1})
            assert record["seconds"] == pytest.approx((i + 1) * 1e-5 + h * 1e-9)
            assert bool(record.get("interpolated")) == (h in (49152, 98304))
    assert len(library.loaded_inputs) == 11


@pytest.mark.parametrize("op", [
    mha(32752), mha(131088), mha(65537), mha(query=14), mha(query=17),
    mha(query=63), mha(layer=0), gdn(query=1), gdn(query=15),
    gdn(query=17), gdn(query=95), gdn(layer=3), gdn(read=-1), gather(index=0),
])
def test_outside_source_scope_delegates_without_new_coverage(bundle, op):
    base = ParametricPriceLibrary()
    base.launch_charge_seconds = 0
    library = CachedQ16Prices(base, *bundle)
    assert library._source_lookup(op, {"tp": 1}) is None
    assert library.lookup(op, {"tp": 1}) == base.lookup(op, {"tp": 1})
    assert not base._prices and not base._attention_obs


def test_layout_geometry_context_and_topology_remain_distinct(bundle):
    library = CachedQ16Prices(PriceLibrary(), *bundle)
    changes = [dict(layouts=[[0, [12288, 2]]]), dict(dtypes=["float16", "float16"]),
               dict(input_shapes=[[16, 12288], [16, 1024]])]
    for change in changes:
        assert library.lookup(dict(mha(), **change))[0] is None
    for key, value in (("has_cached", False), ("num_cached_tokens", [65519]),
                       ("block_tables_shape", [1, 8192]), ("is_prefill", False)):
        op = mha()
        op["context"] = [[k, value if k == key else v] for k, v in op["context"]]
        assert library.lookup(op)[0] is None
    assert library.lookup(mha(), {"tp": 2})[0] is None


def test_fork_relocation_and_inplace_are_only_local_q16_equivalences(bundle):
    base = ParametricPriceLibrary()
    base.launch_charge_seconds = 0
    library = CachedQ16Prices(base, *bundle)
    reference = library.lookup(gdn())[0]["seconds"]
    for read, write in ((2, 3), (0, 0), (1, 7)):
        assert library.lookup(gdn(read=read, write=write))[0]["seconds"] == reference
    assert base.lookup(gdn())[0] is None
    assert library.lookup(gather())[0] is not None


def test_body_retains_baseline_memo_and_host_sync_boundary(bundle):
    class Baseline(PriceLibrary):
        def _body_lookup(self, op, topology, registration, memo):
            memo["calls"] = memo.get("calls", 0) + 1
            return {"seconds": memo["calls"] * 1e-6}, "baseline"

        def host_sync_reason(self, op):
            return "native cached-prefix host sync" if op["name"] == MHA else None

    library = CachedQ16Prices(Baseline(), *bundle)
    ordinary = {"name": "ordinary", "input_shapes": [], "dtypes": [], "scalars": []}
    timing = {}
    seconds, coverage, _ = library.body(
        {"ops": [ordinary, ordinary, mha(49152)], "key": {"topology": [["tp", 1]]}},
        timing=timing)
    assert seconds == pytest.approx(3e-6 + 1e-5 + 49152e-9)
    assert coverage.complete and coverage.interpolated == 1
    assert timing["seconds"] == 3e-6
    assert timing["reason"] == "native cached-prefix host sync"


def test_wrong_bytes_and_launch_charge_are_refused(bundle):
    with pytest.raises(ValueError, match="SHA-256"):
        CachedQ16Prices(PriceLibrary(), bundle[0], "wrong")
    base = PriceLibrary()
    base.launch_charge_seconds = 1e-6
    with pytest.raises(ValueError, match="launch count"):
        CachedQ16Prices(base, *bundle)
    path, digest = bundle
    from pathlib import Path

    price = next(Path(path).parent.glob("*.price.json"))
    price.write_text(price.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256"):
        CachedQ16Prices(PriceLibrary(), path, digest)


def test_prepared_static_segments_are_still_reused(bundle, tmp_path):
    from .test_prepared_plan import add_book, prepared_graph

    base = ParametricPriceLibrary()
    base.launch_charge_seconds = 0
    ordinary = {"name": "ordinary", "input_shapes": [], "dtypes": [], "scalars": []}
    add_book(base, tmp_path, "baseline", [ordinary], [2e-6])
    library = CachedQ16Prices(base, *bundle)
    graph = prepared_graph([ordinary, ordinary, mha(49152), ordinary])
    expected = library.body(dict(graph))
    for _ in range(2):
        assert library.body(graph) == expected
    assert expected[1].complete
    assert library._prepared_plan_prices
