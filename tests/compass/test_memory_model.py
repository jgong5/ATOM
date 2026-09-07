"""Deriving the activation term by walking the graph.

Not how much memory the operators touch -- how much is live at once. That needs
to know which tensor is which, which shapes alone cannot say, so the trace
records which operator produced each input.
"""

import json
import struct

from atom.compass.core.memory_model import (
    graph_pool_bytes, peak_activation_bytes, weight_bytes)


def _write_checkpoint(directory, tied, tensors):
    """A safetensors file with a real header and no tensor data behind it.

    `weight_bytes` reads the header and never the payload, so the bytes the
    header claims are the only ones that need to exist.
    """
    header, offset = {}, 0
    for name, (shape, itemsize) in tensors.items():
        size = itemsize
        for dim in shape:
            size *= dim
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode("utf-8")
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob)
    (directory / "config.json").write_text(
        json.dumps({"tie_word_embeddings": bool(tied)}))


def _op(out, dtype="float32", inputs_from=()):
    return {"name": "aten::x", "input_shapes": [], "output_shapes": [out],
            "dtypes": [dtype], "inputs_from": list(inputs_from),
            "output_aliases": [None]}


def _in_place(out, of, dtype="float32"):
    """An operator writing into the tensor operator `of` produced."""
    return {"name": "aten::x_", "input_shapes": [out], "output_shapes": [out],
            "dtypes": [dtype], "inputs_from": [of], "output_aliases": [of]}


class TestLiveness:
    def test_a_chain_holds_two_tensors_not_all_of_them(self):
        """Each link is dead as soon as the next has read it."""
        graph = {"ops": [_op([1024]), _op([1024], inputs_from=[0]),
                         _op([1024], inputs_from=[1]),
                         _op([1024], inputs_from=[2])]}
        assert peak_activation_bytes(graph) == 2 * 1024 * 4

    def test_a_tensor_read_late_stays_live(self):
        """A residual read at the end of a block is live across the block."""
        graph = {"ops": [_op([1024]), _op([16], inputs_from=[0]),
                         _op([16], inputs_from=[1]),
                         _op([16], inputs_from=[2, 0])]}
        # The 4KB tensor is live throughout. The peak is while an operator
        # runs: its own output exists *and* the input it is still reading, so
        # two of the small ones, not one.
        assert peak_activation_bytes(graph) == 1024 * 4 + 2 * 16 * 4

    def test_weights_are_not_activations(self):
        """An input no operator produced is a weight, and counting it here
        would double it against the weight term."""
        graph = {"ops": [_op([1024], inputs_from=[-1, -1])]}
        assert peak_activation_bytes(graph) == 1024 * 4

    def test_dtype_decides_the_bytes(self):
        small = {"ops": [_op([1024], dtype="bfloat16")]}
        large = {"ops": [_op([1024], dtype="float32")]}
        assert peak_activation_bytes(large) == 2 * peak_activation_bytes(small)


class TestWeights:
    def test_a_checkpoint_that_is_not_there_is_not_guessed(self, tmp_path):
        assert weight_bytes(str(tmp_path)) is None

    def test_a_sharded_checkpoint_reports_its_total(self, tmp_path):
        import json

        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 8_000_000_000}}))
        assert weight_bytes(str(tmp_path)) == 8_000_000_000
        assert weight_bytes(str(tmp_path), tensor_parallel=4) == 2_000_000_000

    def test_a_tied_head_is_stored_but_not_resident(self, tmp_path):
        """The file holds both tensors; the model loads one and points the
        other at it, so counting the file over-counts by an embedding."""
        _write_checkpoint(tmp_path, tied=True, tensors={
            "model.embed_tokens.weight": ([151936, 1024], 2),
            "lm_head.weight": ([151936, 1024], 2),
            "model.layers.0.mlp.gate_proj.weight": ([3072, 1024], 2),
        })
        embedding = 151936 * 1024 * 2
        assert weight_bytes(str(tmp_path)) == embedding + 3072 * 1024 * 2

    def test_an_untied_head_is_counted(self, tmp_path):
        _write_checkpoint(tmp_path, tied=False, tensors={
            "model.embed_tokens.weight": ([1000, 8], 2),
            "lm_head.weight": ([1000, 8], 2),
        })
        assert weight_bytes(str(tmp_path)) == 2 * 1000 * 8 * 2

    def test_tensor_parallelism_shards_the_matrices_and_not_the_norms(
            self, tmp_path):
        """Dividing everything by the world size under-counts by whatever is
        replicated, which is the unsafe direction for a budget."""
        _write_checkpoint(tmp_path, tied=False, tensors={
            "model.layers.0.mlp.gate_proj.weight": ([3072, 1024], 2),
            "model.layers.0.input_layernorm.weight": ([1024], 2),
        })
        matrix, norm = 3072 * 1024 * 2, 1024 * 2
        assert weight_bytes(str(tmp_path), tensor_parallel=2) == matrix // 2 + norm

    def test_headers_that_cannot_be_read_fall_back_to_the_size_on_disk(
            self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"not a checkpoint")
        assert weight_bytes(str(tmp_path)) == len(b"not a checkpoint")


class TestGraphPool:
    def test_eager_captures_nothing_and_so_holds_nothing(self):
        assert graph_pool_bytes(1 << 30, enforce_eager=True) == 0

    def test_manual_capture_is_a_fraction_of_the_activations(self):
        """So this term composes with the activation term, and inherits its
        error -- which is why they are checked separately."""
        assert graph_pool_bytes(1000) == 200

    def test_piecewise_is_geometry_over_the_buckets_that_fit(self):
        pool = graph_pool_bytes(0, piecewise=True, hidden_size=1024,
                                num_hidden_layers=28, dtype_bytes=2,
                                capture_sizes=(1, 2, 4),
                                total_bytes=1 << 40, utilization=0.9)
        per_token = 1024 * 2 * 28 * 2.8
        assert pool == int(per_token * (1 + 2 + 4))

    def test_a_bucket_over_the_token_budget_is_never_captured(self):
        """The capture loop skips it, so reserving for it would only shrink
        the KV cache."""
        with_big = graph_pool_bytes(0, piecewise=True, hidden_size=8,
                                    num_hidden_layers=1, capture_sizes=(1, 9999),
                                    max_num_batched_tokens=64,
                                    total_bytes=1 << 40, utilization=0.9)
        assert with_big == int(8 * 2 * 1 * 2.8 * 1)

    def test_the_reservation_stops_at_a_fraction_of_the_budget(self):
        """A long capture list must not starve the cache it is reserved from."""
        pool = graph_pool_bytes(0, piecewise=True, hidden_size=1 << 20,
                                num_hidden_layers=64, capture_sizes=(1, 2, 4, 8),
                                total_bytes=1 << 30, utilization=1.0)
        assert pool < graph_pool_bytes(
            0, piecewise=True, hidden_size=1 << 20, num_hidden_layers=64,
            capture_sizes=(1, 2, 4, 8), total_bytes=1 << 40, utilization=1.0)


class TestInPlaceOperators:
    """An operator that writes into its input allocates nothing.

    Counting its output as a fresh tensor adds one to the live set that was
    never there; treating it as a consumer frees a tensor still being read.
    """

    def test_writing_into_an_input_adds_nothing(self):
        graph = {"ops": [_op([1024]), _in_place([1024], of=0)]}
        assert peak_activation_bytes(graph) == 1024 * 4

    def test_the_tensor_stays_live_for_the_readers_of_the_in_place_result(self):
        """`gemm -> all_reduce_ -> read` is one tensor across all three."""
        graph = {"ops": [_op([1024]), _in_place([1024], of=0),
                         _op([8], inputs_from=[1]), _op([8], inputs_from=[2])]}
        # The 4KB tensor is live until operator 2 has read it, so the peak is
        # it plus operator 2's own output.
        assert peak_activation_bytes(graph) == 1024 * 4 + 8 * 4

    def test_writing_into_something_from_before_the_step_is_not_an_activation(
            self):
        """A KV cache written in place is not this step's memory."""
        graph = {"ops": [_in_place([4096], of=-1)]}
        assert peak_activation_bytes(graph) == 0

    def test_a_graph_without_the_field_walks_as_it_used_to(self):
        """Records predating the field have no aliases and must still read."""
        graph = {"ops": [_op([1024]), _op([1024], inputs_from=[0])]}
        assert peak_activation_bytes(graph) == 2 * 1024 * 4

    def test_an_operator_that_both_allocates_and_writes_keeps_what_it_made(self):
        """Attention returns a tensor *and* fills the KV cache. Redirecting it
        to the cache would drop the tensor it allocated from the live set."""
        both = {"name": "aiter::attention", "input_shapes": [], "dtypes": ["float32"],
                "output_shapes": [[4096], [1024]], "inputs_from": [-1],
                "output_aliases": [-1, None]}
        assert peak_activation_bytes({"ops": [both]}) == 1024 * 4
