"""Per-regime calibration selection cannot be bypassed by legacy exact rows."""
import hashlib
import json

import pytest

from atom.compass.core.cost.families import attention as A
from atom.compass.runtime.microbench import signature_of
from atom.compass.runtime.source_oracle import _price_library, _DEFAULT_GAP_RATIO

NAME = "unified.decode.paged_gluon_order"
SCOPE = {"attention_backend": "paged_gluon", "sliding_window": -1,
         "num_kv_heads": 4, "compute_units": 80, "kv_cache_dtype": "bfloat16",
         "kv_cache_layout": "NHD", "kv_cache_block_size": 16}


def op_for(contexts, *, prefill=False, cached=False):
    count = len(contexts)
    return {"name": A.UNIFIED,
            "input_shapes": [[count, 6144], [count, 1024], [count, 1024]],
            "dtypes": ["bfloat16"] * 3,
            "context": [["context_lens", contexts], ["cu_seqlens_q", list(range(count + 1))],
                        ["max_seqlen_q", 1], ["max_seqlen_k", max(contexts)],
                        ["is_prefill", prefill], ["has_cached", cached]]}


def file_pair(tmp_path, label, op, seconds, rotation, kernel="decode_kernel"):
    signature = signature_of(op)
    graph, price = tmp_path / (label + '.graph.json'), tmp_path / (label + '.price.json')
    graph.write_text(json.dumps({"ops": [op]}))
    price.write_text(json.dumps({"provenance": {"attention_scope": SCOPE},
                                 "prices": {signature: {"signature": signature, "name": op['name'],
                                     "seconds": seconds, "kernels": {kernel: seconds},
                                     "cache": "graph", "kv_regions": rotation}}}))
    return str(price), str(graph)


def files(tmp_path):
    entries = []
    for index, (rows, context) in enumerate(((1, 4096), (2, 8192), (8, 4096),
                                             (16, 4096), (32, 4096), (32, 8192))):
        op = op_for([context] * rows)
        values = A.features_for(A.REGIMES[NAME], A.structure_of(op), SCOPE)
        # Synthetic source timings serve only to make the routing test's law
        # identifiable; accuracy is established by the real frozen holdouts.
        seconds = 25e-6 + max(7e-6 * values[1], 13e-6 * values[2])
        entries.append(file_pair(tmp_path, str(index), op, seconds, 2))
    asked = op_for([4096] * 16)
    entries.insert(0, file_pair(tmp_path, "legacy_exact", asked, 0.9, 8))
    cold, cached = op_for([1], prefill=True), op_for([32], prefill=True, cached=True)
    entries.append(file_pair(tmp_path, "cold", cold, .0032, 32, "cold_kernel"))
    entries.append(file_pair(tmp_path, "cached", cached, .0008, 8, "cached_kernel"))
    return entries, asked, cold, cached


def build(entries, selector):
    return _price_library(entries, _DEFAULT_GAP_RATIO, attention_scope=SCOPE,
                           attention_treatments=selector)


def test_selected_decode_bypasses_legacy_exact_and_preserves_prefill(tmp_path, monkeypatch):
    monkeypatch.setitem(A.DECODE_KERNELS, "paged_gluon", NAME)
    entries, asked, cold, cached = files(tmp_path)
    baseline = build(entries, None)
    legacy, _ = baseline.lookup(asked)
    assert legacy['seconds'] == .9
    selected = build(entries, {NAME: {"kv_regions": 2, "cache": "graph"}})
    answer, origin = selected.lookup(asked)
    assert answer is not None, origin
    assert answer['interpolated'] is True
    assert answer['interpolation']['regime'] == NAME
    assert answer['seconds'] < .01
    for op, expected in ((cold, .0032), (cached, .0008)):
        assert selected.lookup(op)[0]['seconds'] == baseline.lookup(op)[0]['seconds'] == expected


@pytest.mark.parametrize("case", ["absent", "ambiguous"])
def test_unmatched_or_ambiguous_selection_refuses_even_with_exact_legacy(tmp_path, monkeypatch, case):
    monkeypatch.setitem(A.DECODE_KERNELS, "paged_gluon", NAME)
    entries, asked, _cold, _cached = files(tmp_path)
    rotation = 4 if case == "absent" else 2
    if case == "ambiguous":
        entries.append(file_pair(tmp_path, "other_policy", asked, .0001, 2, "different_kernel"))
    library = build(entries, {NAME: {"kv_regions": rotation}})
    answer, reason = library.lookup(asked)
    assert answer is None
    assert "measurement_treatment" in reason


def test_selection_file_is_digested_and_reported(tmp_path, monkeypatch):
    monkeypatch.setitem(A.DECODE_KERNELS, "paged_gluon", NAME)
    entries, asked, _cold, _cached = files(tmp_path)
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps({NAME: {"kv_regions": 2}}))
    raw = path.read_bytes()
    library = build(entries, str(path))
    assert library.lookup(asked)[0] is not None
    records = [r for r in library.loaded_inputs if r.role == "oracle.attention_treatments"]
    assert len(records) == 1 and records[0].sha256 == hashlib.sha256(raw).hexdigest()
    assert library.attention_coverage()['requested_treatments'][NAME]['kv_regions'] == 2


def test_recorded_file_hash_is_evidence_but_not_an_import_attestation():
    from atom.compass.core.cost.families.adapter import _acquisition_policy
    module = "atom.compass.runtime.microbench"
    digest = "a" * 64
    legacy = {"provenance": {"collector": {"hashes": {
        "sources": {"atom/compass/runtime/microbench.py": digest}}}}}
    imported = {"provenance": {"collector": {"source_identity": {
        "modules": {module: {"sha256": digest}}}}}}
    assert _acquisition_policy(legacy) == ("recorded_source_file", module, digest)
    assert _acquisition_policy(imported) == (module, digest)
    assert _acquisition_policy(legacy) != _acquisition_policy(imported)
    assert _acquisition_policy({}) == ("unevidenced",)
