"""The metadata recipe, checked against a forward that actually ran.

A recipe that computes attention's metadata from a batch description is only
worth having if it computes what the engine computed. The 27B capture recorded
both: `provenance.shape` says what the batch was, and every attention operator
carries the context the live forward held. So the test is a replay -- describe
that batch, derive the metadata, and compare field by field with what was
recorded.

The capture is `compass_ops/silu27_graph.tp0.json`, a 4-request decode at
context 66 with 1 token each, block size 16, max_model_len 4096. It is not
required to be present: where it is, the comparison runs against it; where it
is not, the same values are asserted from the constants below, which were read
off it once and are quoted in `test_the_recorded_case_is_what_we_think`.
"""

import json
import os

import pytest

from atom.compass.runtime.batch_spec import (BatchSpec, allocate_blocks,
                                             model_inputs)

CAPTURE = "compass_ops/silu27_graph.tp0.json"

#: Read off the capture named above. Quoted rather than only computed, so a
#: recipe change that silently agrees with itself still fails here.
RECORDED_ATTENTION = {
    "context_lens": [66, 66, 66, 66],
    "slot_mapping": [257, 273, 289, 305],
    "cu_seqlens_q": [0, 1, 2, 3, 4],
    "cu_seqlens_k": None,
    "max_seqlen_q": 1,
    "max_seqlen_k": 66,
    "min_seqlen_q": 0,
    "has_cached": False,
    "state": "prefill_native",
    "is_prefill": False,
    "positions": [65] * 12,
    "block_tables_shape": [4, 256],
    "block_tables": [0, 1, 2, 3, 16, 4, 5, 6, 7, 17,
                     8, 9, 10, 11, 18, 12, 13, 14, 15, 19],
}
RECORDED_GDN = {
    "num_prefills": 0, "num_prefill_tokens": 0,
    "num_decodes": 4, "num_decode_tokens": 4,
    "num_spec_decodes": 0, "num_spec_decode_tokens": 0,
    "num_actual_tokens": 4, "replayssm": False,
    "non_spec_query_start_loc": [[0, 1, 2, 3, 4], "int32"],
    "non_spec_state_indices_tensor": [[0, 1, 2, 3], "int32"],
    "non_spec_state_indices_in_tensor": [[0, 1, 2, 3], "int32"],
}


def _captured_decode() -> BatchSpec:
    """The capture's batch, as its own provenance describes it."""
    return BatchSpec(
        kind="decode",
        query_lens=(1, 1, 1, 1),
        context_lens=(66, 66, 66, 66),
        block_size=16,
        max_model_len=4096,
        capture_bucket=None,
        # 64 tokens at admission, two generated since: four prompt blocks in
        # request order, then a fifth apiece in a second round.
        prompt_lens=(64, 64, 64, 64),
        # MRoPE lays positions out as [3, N], which is why the recorded tensor
        # holds 12 values for a batch of 4.
        position_rows=3,
    )


def _from_capture(name: str):
    """The context the capture recorded for `name`, or None if it is absent."""
    if not os.path.exists(CAPTURE):
        return None
    with open(CAPTURE, encoding="utf-8") as fh:
        blob = json.load(fh)
    for op in blob["ops"]:
        if op["name"] == name and op.get("context"):
            return {k: v for k, v in (tuple(x) for x in op["context"])}
    return None


class TestTheRecipeReproducesACaptureFieldByField:
    """Every value attention read, recomputed from the batch description."""

    @pytest.mark.parametrize("field", sorted(RECORDED_ATTENTION))
    def test_attention(self, field):
        derived = dict(_captured_decode().attention_context())
        assert derived[field] == RECORDED_ATTENTION[field], field

    def test_and_nothing_else(self):
        """No field invented, none dropped."""
        derived = dict(_captured_decode().attention_context())
        assert sorted(derived) == sorted(RECORDED_ATTENTION)

    @pytest.mark.parametrize("field", sorted(RECORDED_GDN))
    def test_deltanet(self, field):
        derived = dict(_captured_decode().gdn_context())
        assert derived[field] == RECORDED_GDN[field], field

    def test_the_recorded_case_is_what_we_think(self):
        """Guard the quoted constants against the artifact itself.

        Skipped where the capture is not on disk -- the values above still
        stand, they are simply no longer cross-checked.
        """
        recorded = _from_capture("aiter::unified_attention_with_output_base")
        if recorded is None:
            pytest.skip(f"{CAPTURE} not present")
        assert recorded == RECORDED_ATTENTION

    def test_the_recorded_deltanet_case_is_too(self):
        recorded = _from_capture("aiter::linear_attention_with_output_base")
        if recorded is None:
            pytest.skip(f"{CAPTURE} not present")
        assert recorded == RECORDED_GDN


class TestWhichBlocksARequestHolds:
    """Block ids are the manager's history, so the policy is named and checked."""

    def test_rounds_reproduces_the_capture(self):
        """Four 64-token prompts, then one growth block each, in request order."""
        assert allocate_blocks([64] * 4, [66] * 4, 16, "rounds") == [
            [0, 1, 2, 3, 16], [4, 5, 6, 7, 17],
            [8, 9, 10, 11, 18], [12, 13, 14, 15, 19]]

    def test_packed_does_not(self):
        """Offered as an alternative, not as an equivalent."""
        assert allocate_blocks([64] * 4, [66] * 4, 16, "packed") == [
            [0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]

    def test_a_prefill_allocates_in_one_round(self):
        """Its whole context is its prompt, so there is no growth round."""
        assert allocate_blocks([32, 48], [32, 48], 16, "rounds") == [
            [0, 1], [2, 3, 4]]

    def test_a_request_that_grew_does_not(self):
        """The same contexts, reached by generating rather than by prompting."""
        assert allocate_blocks([0, 0], [32, 48], 16, "rounds") == [
            [0, 2], [1, 3, 4]]

    def test_an_unknown_policy_is_not_guessed_at(self):
        with pytest.raises(ValueError, match="policy"):
            allocate_blocks([16], [16], 16, "whatever-fits")


class TestPrefillIsNotDecodeWithTheSameShapes:
    """The distinction the token count could not make."""

    def _prefill(self, **kw):
        return BatchSpec(kind="prefill", query_lens=(4,), context_lens=(4,),
                         block_size=16, max_model_len=4096, **kw)

    def test_a_native_prefill_reads_no_cached_kv(self):
        spec = self._prefill()
        assert spec.has_cached is False
        derived = dict(spec.attention_context())
        assert derived["state"] == "prefill_native"
        assert derived["is_prefill"] is True
        assert derived["cu_seqlens_k"] == [0, 4]
        assert "total_kv" not in derived

    def test_a_chunked_prefill_does_and_says_so(self):
        """The three fields the gather sizes itself from, and only here."""
        spec = BatchSpec(kind="prefill", query_lens=(4,), context_lens=(70,),
                         block_size=16, max_model_len=4096)
        derived = dict(spec.attention_context())
        assert derived["state"] == "prefill_prefix"
        assert derived["has_cached"] is True
        assert derived["total_kv"] == 70
        assert derived["num_cached_tokens"] == [66]
        assert derived["seq_starts"] == [0]

    def test_four_tokens_one_sequence_is_not_four_decodes(self):
        """The claim the old token-count derivation could not distinguish."""
        body = self._prefill()
        decodes = BatchSpec(kind="decode", query_lens=(1, 1, 1, 1),
                            context_lens=(66, 66, 66, 66), block_size=16,
                            max_model_len=4096)
        assert body.num_tokens == decodes.num_tokens == 4
        a, b = dict(body.attention_context()), dict(decodes.attention_context())
        assert a["max_seqlen_k"] != b["max_seqlen_k"]
        assert a["is_prefill"] != b["is_prefill"]
        assert a["cu_seqlens_q"] != b["cu_seqlens_q"]

    def test_deltanet_counts_prefills_where_attention_sets_a_flag(self):
        derived = dict(self._prefill().gdn_context())
        assert derived["num_prefills"] == 1
        assert derived["num_prefill_tokens"] == 4
        assert derived["num_decodes"] == 0


class TestWhatCanBeCheckedWithoutADevice:
    """A spec that describes a step no engine could run must not price."""

    def _spec(self, **kw):
        base = dict(kind="decode", query_lens=(1,), context_lens=(66,),
                    block_size=16, max_model_len=4096)
        base.update(kw)
        return BatchSpec(**base)

    def test_a_context_past_the_model_is_refused(self):
        with pytest.raises(ValueError, match="max_model_len"):
            self._spec(context_lens=(8192,)).validate()

    def test_a_query_longer_than_its_context_is_refused(self):
        with pytest.raises(ValueError, match="into a context"):
            BatchSpec(kind="prefill", query_lens=(70,), context_lens=(66,),
                      block_size=16, max_model_len=4096).validate()

    def test_a_decode_computing_two_tokens_is_refused(self):
        """Unless it declares the speculative steps that would explain it."""
        with pytest.raises(ValueError, match="num_spec_step"):
            self._spec(query_lens=(2,)).validate()
        self._spec(query_lens=(2,), num_spec_step=1).validate()

    def test_a_table_too_short_for_the_context_is_refused(self):
        with pytest.raises(ValueError, match="needs 5 blocks"):
            self._spec(block_tables=((0, 1),)).validate()

    def test_a_table_wider_than_the_model_allows_is_refused(self):
        """A 64-token model has a four-block row, whatever the table says."""
        with pytest.raises(ValueError, match="a table row is"):
            BatchSpec(kind="decode", query_lens=(1,), context_lens=(64,),
                      block_size=16, max_model_len=64,
                      block_tables=(tuple(range(5)),)).validate()

    def test_a_bucket_narrower_than_the_batch_is_refused(self):
        with pytest.raises(ValueError, match="capture bucket"):
            self._spec(query_lens=(1, 1), context_lens=(66, 66),
                       capture_bucket=1).validate()

    def test_the_bounds_hold_without_importing_torch(self):
        """Validation is arithmetic, so it runs where derivation runs."""
        import sys

        assert "atom.compass.runtime.batch_spec" in sys.modules
        module = sys.modules["atom.compass.runtime.batch_spec"]
        assert not hasattr(module, "torch")


class TestSlotsAndPositionsFollowTheQuery:
    """Where each computed token is written, and what position it holds."""

    def test_a_decode_writes_one_slot_per_request(self):
        spec = BatchSpec(kind="decode", query_lens=(1, 1),
                         context_lens=(17, 33), block_size=16,
                         max_model_len=4096, block_tables=((7, 9), (2, 3, 5)))
        derived = dict(spec.attention_context())
        # token 16 of request 0 is block index 1 -> id 9, offset 0
        # token 32 of request 1 is block index 2 -> id 5, offset 0
        assert derived["slot_mapping"] == [9 * 16 + 0, 5 * 16 + 0]
        assert derived["positions"] == [16, 32]

    def test_a_prefill_writes_every_token_it_computes(self):
        spec = BatchSpec(kind="prefill", query_lens=(3,), context_lens=(3,),
                         block_size=16, max_model_len=4096,
                         block_tables=((4,),))
        derived = dict(spec.attention_context())
        assert derived["slot_mapping"] == [64, 65, 66]
        assert derived["positions"] == [0, 1, 2]

    def test_a_chunked_prefill_writes_only_the_new_tokens(self):
        spec = BatchSpec(kind="prefill", query_lens=(2,), context_lens=(18,),
                         block_size=16, max_model_len=4096,
                         block_tables=((4, 6),))
        derived = dict(spec.attention_context())
        # positions 16 and 17: block index 1 -> id 6, offsets 0 and 1
        assert derived["slot_mapping"] == [96, 97]
        assert derived["positions"] == [16, 17]

    def test_the_table_is_cut_to_the_longest_context_not_the_row_width(self):
        """A 4096-token model has a 256-wide row and a 66-token decode reads 5.

        Keeping the used prefix is what stops the artifact growing with the
        model's maximum context; keeping the shape is what preserves the row
        stride the kernel indexes with.
        """
        derived = dict(_captured_decode().attention_context())
        assert derived["block_tables_shape"] == [4, 256]
        assert len(derived["block_tables"]) == 4 * 5


class TestReadingOneOffDisk:
    """A spec is a file the derivation is given, so the file is the contract."""

    def test_json_lists_become_the_tuples_the_spec_compares_by(self):
        spec = BatchSpec.from_dict({
            "kind": "decode", "query_lens": [1, 1], "context_lens": [66, 66],
            "block_size": 16, "max_model_len": 4096})
        assert spec.query_lens == (1, 1)
        assert spec == BatchSpec(kind="decode", query_lens=(1, 1),
                                 context_lens=(66, 66), block_size=16,
                                 max_model_len=4096)

    def test_a_misspelled_field_is_an_error_not_a_default(self):
        """`prompt_len` would silently derive a different block allocation."""
        with pytest.raises(ValueError, match="prompt_len"):
            BatchSpec.from_dict({
                "kind": "decode", "query_lens": [1], "context_lens": [66],
                "block_size": 16, "max_model_len": 4096, "prompt_len": [64]})

    def test_an_impossible_batch_is_refused_at_read_time(self):
        with pytest.raises(ValueError):
            BatchSpec.from_dict({
                "kind": "decode", "query_lens": [4], "context_lens": [66],
                "block_size": 16, "max_model_len": 4096})

    def test_a_round_trip_through_json_is_the_same_batch(self, tmp_path):
        spec = _captured_decode()
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec.to_dict()))
        back = BatchSpec.load(str(path))
        assert back.attention_context() == spec.attention_context()

    def test_writing_it_out_makes_the_derived_block_table_explicit(self):
        """The policy is reproducible only if its output is there to check."""
        raw = _captured_decode().to_dict()
        assert raw["block_tables"] == _captured_decode().tables()
        assert BatchSpec.from_dict(raw).tables() == _captured_decode().tables()

    def test_the_shipped_spec_is_the_batch_the_capture_ran(self):
        """agent_scratch is not shipped; this one lives beside the code."""
        path = os.path.join(os.path.dirname(__file__), "batch_specs",
                            "decode4_c66.json")
        spec = BatchSpec.load(path)
        assert spec.attention_context() == _captured_decode().attention_context()


class TestWhatTheModelIsHanded:
    """`model_inputs` is the other half: the tensors, not the metadata."""

    def test_decode_positions_are_the_last_token_of_each_request(self):
        torch = pytest.importorskip("torch")
        spec = BatchSpec(kind="decode", query_lens=(1, 1), context_lens=(66, 40),
                         block_size=16, max_model_len=4096)
        ids, pos = model_inputs(spec, device="cpu")
        assert ids.dtype == torch.int32 and ids.shape == (2,)
        assert pos.dtype == torch.int64
        assert pos.tolist() == [65, 39]

    def test_mrope_is_handed_three_rows_not_a_flat_run(self):
        """`model_runner._mrope_positions_view` builds [3, N]; RoPE indexes it,

        so the shape reaches the graph and a flat tensor derives a different one.
        """
        pytest.importorskip("torch")
        _, pos = model_inputs(_captured_decode(), device="cpu")
        assert pos.shape == (3, 4)
        assert pos.tolist() == [[65, 65, 65, 65]] * 3

    def test_one_row_stays_flat(self):
        pytest.importorskip("torch")
        spec = BatchSpec(kind="prefill", query_lens=(3,), context_lens=(3,),
                         block_size=16, max_model_len=4096)
        _, pos = model_inputs(spec, device="cpu")
        assert pos.shape == (3,)
        assert pos.tolist() == [0, 1, 2]

    def test_the_flattened_rows_are_what_the_context_records(self):
        """One tensor, two readers: the model gets [3, N] and the forward

        context records the same buffer flat. They cannot disagree.
        """
        pytest.importorskip("torch")
        spec = _captured_decode()
        _, pos = model_inputs(spec, device="cpu")
        assert pos.flatten().tolist() == dict(spec.attention_context())["positions"]
