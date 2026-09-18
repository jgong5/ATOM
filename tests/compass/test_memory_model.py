"""Deriving the activation term by walking the graph.

Not how much memory the operators touch -- how much is live at once. That needs
to know which tensor is which, which shapes alone cannot say, so the trace
records which operator produced each input.
"""

import json
import struct

import pytest

from atom.compass.core.memory_model import (
    DEFAULT_NON_TORCH, DEFAULT_PERSISTENT, DEFAULT_POOL_FLOOR,
    activation_bytes_at, activation_curve, graph_pool_bytes,
    load_residue_bytes, measured_graph_pool_bytes, modelled_readings,
    non_torch_bytes, peak_activation_bytes, scratch_bytes_per_token,
    weight_bytes)


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


def _op(out, dtype="float32", inputs_from=(), dies_at=-1):
    return {"name": "aten::x", "input_shapes": [], "output_shapes": [out],
            "dtypes": [dtype], "inputs_from": list(inputs_from),
            "output_aliases": [None], "dies_at": [dies_at]}


def _in_place(out, of, dtype="float32", dies_at=-1):
    """An operator writing into the tensor operator `of` produced."""
    return {"name": "aten::x_", "input_shapes": [out], "output_shapes": [out],
            "dtypes": [dtype], "inputs_from": [of], "output_aliases": [of],
            "dies_at": [dies_at]}


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
                "output_aliases": [-1, None], "dies_at": [-1, -1]}
        assert peak_activation_bytes({"ops": [both]}) == 1024 * 4


class TestDeathsThatWereObservedRatherThanGuessed:
    """Last-read is not when a tensor dies, and is wrong in both directions.

    A local held across a block outlives every read of it; and a producer map
    keyed on storage address credits a reused address to whatever held it
    before, which keeps a dead tensor in the live set to the end of the step.
    """

    def test_a_recorded_death_beats_the_last_read(self):
        """A residual read once early and held to the end of the block.

        Last-read frees it at the one operator that looked at it; the
        recording says it was still there three operators later, and it was.
        """
        chain = [_op([16], inputs_from=[0]), _op([16], inputs_from=[1]),
                 _op([16], inputs_from=[2])]
        recorded = {"ops": [_op([1024], dies_at=3)] + chain}
        guessed = {"ops": [_op([1024])] + [dict(o, dies_at=[-1]) for o in chain]}
        assert peak_activation_bytes(recorded) == 1024 * 4 + 3 * 16 * 4
        assert peak_activation_bytes(guessed) < peak_activation_bytes(recorded)

    def test_an_output_with_no_recorded_death_outlived_the_step(self):
        """The opposite default to the last-read rule, and it must not be
        confused with it: nothing observed it dying, so it did not."""
        graph = {"ops": [_op([1024], dies_at=1), _op([1024]), _op([16])]}
        # Operator 0 goes at operator 1 because that was seen; operator 1 was
        # never seen to go, so it is still there at the end.
        assert activation_curve(graph)[-1] == 1024 * 4 + 16 * 4

    def test_a_graph_with_no_deaths_at_all_falls_back_to_last_read(self):
        """A record written before deaths were observed still walks."""
        graph = {"ops": [_op([1024]), _op([1024], inputs_from=[0]),
                         _op([1024], inputs_from=[1])]}
        assert peak_activation_bytes(graph) == 2 * 1024 * 4

    def test_an_in_place_operator_dies_with_the_tensor_it_wrote_into(self):
        graph = {"ops": [_op([1024], dies_at=2), _in_place([1024], of=0),
                         _op([16], inputs_from=[1])]}
        assert peak_activation_bytes(graph) == 1024 * 4 + 16 * 4

    def test_two_outputs_of_one_operator_can_have_different_lives(self):
        """A fused add-and-norm returns the normed activation and the new
        residual. The first dies into the next gemm; the second carries to the
        end of the block. One death for the pair holds an extra tensor per
        layer -- at TP=4 that was 36% of the term."""
        fused = {"name": "aiter::rmsnorm2d_fwd_with_add_", "input_shapes": [],
                 "dtypes": ["float32"], "output_shapes": [[1024], [1024]],
                 "inputs_from": [], "output_aliases": [None, None],
                 "dies_at": [1, 3]}
        graph = {"ops": [fused, _op([16]), _op([16]), _op([16])]}
        # At operator 2 only the residual survives, plus what has run since.
        assert activation_curve(graph)[2] == 1024 * 4 + 2 * 16 * 4
        assert peak_activation_bytes(graph) == 2 * 1024 * 4 + 16 * 4


class TestBuffersNoOperatorDeclared:
    """`torch.empty` inside a custom operator never reaches a dispatch tracer:
    re-entering `func` from `__torch_dispatch__` runs below the mode. The
    buffer is real and is where the high-water mark sits, so the trace records
    the destination of an out-variant as that operator's output."""

    def test_a_destination_recorded_as_an_output_is_counted(self):
        out_variant = {"name": "aiter::silu_and_mul", "dtypes": ["float32"],
                       "input_shapes": [[1024], [2048]],
                       "output_shapes": [[1024]], "inputs_from": [-1, 0],
                       "output_aliases": [None], "dies_at": [-1]}
        graph = {"ops": [_op([2048]), out_variant]}
        assert peak_activation_bytes(graph) == 2048 * 4 + 1024 * 4

    def test_a_destination_that_was_already_someones_output_is_not_doubled(self):
        """When the trace did see the destination allocated, the destination
        belongs to whoever allocated it and the out-variant adds nothing."""
        writes_into = {"name": "aiter::silu_and_mul", "dtypes": ["float32"],
                       "input_shapes": [[1024]], "output_shapes": [[1024]],
                       "inputs_from": [0], "output_aliases": [0],
                       "dies_at": [-1]}
        graph = {"ops": [_op([1024]), writes_into]}
        assert peak_activation_bytes(graph) == 1024 * 4


class TestWhatTheCollectivesTake:
    """`non_torch` is `(total - free) - reserved`, and `total - free` is
    device-wide -- so a neighbour is charged to this configuration. Measured
    per width on one box; a table, not a law, and calibratable per deployment
    for the same reason the overhead constant is."""

    def test_a_single_rank_holds_no_collective_pools(self):
        assert load_residue_bytes(1) < non_torch_bytes(1)
        assert load_residue_bytes(1) < 8 * (1 << 20)

    def test_the_pools_appear_at_the_first_width_above_one(self):
        assert load_residue_bytes(2) > 2000 * (1 << 20)
        assert non_torch_bytes(2) > non_torch_bytes(1)

    def test_the_residue_is_flat_in_width(self):
        """1.1 / 2069 / 2069 / 2068 MiB at widths 1, 2, 4 and 8."""
        assert load_residue_bytes(2) == load_residue_bytes(4)
        assert load_residue_bytes(4) == load_residue_bytes(8)

    def test_a_width_between_entries_takes_the_widest_below_it(self):
        assert non_torch_bytes(3) == non_torch_bytes(2)
        assert non_torch_bytes(6) == non_torch_bytes(4)

    def test_calibration_replaces_the_table(self):
        """The constants do not transfer between boxes, so a deployment
        measures its own rather than trusting these."""
        mine = {"non_torch": {1: 7, 2: 11}, "load_residue": {1: 3, 2: 5}}
        assert non_torch_bytes(2, mine) == 11
        assert load_residue_bytes(2, mine) == 5

    def test_the_model_headroom_is_only_on_the_defaults(self):
        """Carried because the 27B sat 266 MiB above the 0.6B at every width
        and two models cannot say what that is a function of. A calibration
        measured the model in question, so it needs no headroom."""
        assert non_torch_bytes(2) > DEFAULT_NON_TORCH[2]
        assert non_torch_bytes(2, {"non_torch": {2: DEFAULT_NON_TORCH[2]}}) \
            == DEFAULT_NON_TORCH[2]


class TestSizingWithoutADevice:
    """The five readings `get_num_blocks` needs, none of them measured. This is
    what makes a configuration nobody has run sizable."""

    def _readings(self, **over):
        fields = dict(total_bytes=200 << 30, world_size=1, parameters=1 << 30,
                      buffers=1 << 20, activation_bytes=1 << 27)
        fields.update(over)
        return modelled_readings(**fields)

    def test_peak_torch_is_every_term_that_goes_through_the_allocator(self):
        got = self._readings(world_size=1)
        assert got["peak_torch"] == (
            (1 << 30) + (1 << 20) + load_residue_bytes(1)
            + DEFAULT_PERSISTENT + (1 << 27))

    def test_free_is_a_clean_box_not_what_the_neighbours_left(self):
        """The recorded `free` is an accident of scheduling, which is why a
        record in which it bound is refused. Deriving it removes the accident."""
        got = self._readings()
        assert got["free"] == got["total"] - got["peak_torch"] - got["non_torch"]

    def test_a_wider_configuration_pays_for_the_collective_pools(self):
        narrow, wide = self._readings(world_size=1), self._readings(world_size=2)
        assert wide["non_torch"] > narrow["non_torch"]
        assert wide["peak_torch"] > narrow["peak_torch"]

    def test_eager_reserves_no_graph_pool(self):
        assert self._readings(enforce_eager=True)["cudagraph_overhead"] == 0

    def test_a_card_too_small_to_hold_the_model_has_no_free_memory(self):
        """Reported as zero rather than as a negative number, which the
        engine's `min(budget, free)` would read as a very large budget."""
        assert self._readings(total_bytes=1 << 20)["free"] == 0


class TestScalingToAShapeNobodyTraced:
    def test_the_peak_is_linear_in_tokens(self):
        graph = {"key": {"batch_signature": [100]},
                 "ops": [_op([1000], dtype="bfloat16")]}
        assert activation_bytes_at(graph, 200) == 2 * peak_activation_bytes(graph)

    def test_a_graph_that_names_no_shape_is_taken_as_it_stands(self):
        graph = {"ops": [_op([1000], dtype="bfloat16")]}
        assert activation_bytes_at(graph, 200) == peak_activation_bytes(graph)


class TestWhatCaptureActuallyCosts:
    """Two different questions. `graph_pool_bytes` mirrors the engine's
    estimator, which is the number that reserves the memory and so is what a
    modelled budget must reproduce. `measured_graph_pool_bytes` predicts the
    cost. They disagree by 4-19x, and the engine's is blind to the ladder."""

    LADDER = (1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512)

    def test_the_pool_grows_with_the_ladder_where_the_estimate_does_not(self):
        short = measured_graph_pool_bytes((1, 2, 4, 8, 16))
        long = measured_graph_pool_bytes(self.LADDER)
        assert long > 3 * short
        # The engine's estimator takes the peak activations, which belong to
        # the warmup shape; the ladder does not enter it.
        assert graph_pool_bytes(1 << 27) == graph_pool_bytes(1 << 27)

    def test_the_floor_dominates_a_short_ladder(self):
        """At the shortest ladder measured, 87% of the pool was the floor."""
        tiny = measured_graph_pool_bytes((1,))
        assert tiny > 0.9 * DEFAULT_POOL_FLOOR

    def test_capturing_nothing_costs_nothing(self):
        assert measured_graph_pool_bytes(()) == 0
        assert measured_graph_pool_bytes(self.LADDER, enforce_eager=True) == 0

    def test_above_width_one_the_ladder_stops_mattering(self):
        """Measured to the byte across three widths and three ladders: the
        allocated delta was 79692800 every time, so the graphs are not pinning
        sharded activations -- with tensor parallelism the intermediates go
        through the registered collective buffer, outside torch."""
        for width in (2, 4, 8):
            assert (measured_graph_pool_bytes((1,), world_size=width)
                    == measured_graph_pool_bytes(self.LADDER, world_size=width))

    def test_a_wider_rank_holds_less_of_it(self):
        assert (measured_graph_pool_bytes(self.LADDER, world_size=2)
                < measured_graph_pool_bytes(self.LADDER, world_size=1))

    def test_calibration_replaces_the_constants(self):
        mine = {"graph_pool": {"floor": 10, "per_token": 1, "sharded": 3}}
        assert measured_graph_pool_bytes((5,), calibration=mine) == 15
        assert measured_graph_pool_bytes((5,), world_size=4,
                                         calibration=mine) == 3

    def test_it_reproduces_the_ladders_it_was_fitted_on(self):
        """100.0 MiB measured at sum=31 and 402.0 at sum=1071."""
        M = 1 << 20
        assert abs(measured_graph_pool_bytes((1, 2, 4, 8, 16)) / M - 100.0) < 8
        assert abs(measured_graph_pool_bytes(self.LADDER) / M - 402.0) < 24


class TestWhatTheTracerCannotSee:
    """`torch.empty` inside a custom operator never crosses the dispatcher. For
    an out-variant the destination is recoverable and is recovered; for an
    operator that *returns* a tensor and also allocates internal scratch,
    nothing in the graph says the scratch exists. The 0.6B has almost none
    (0.1 KB/token); the hybrid 27B has 39.6, enough to put the walk 37% under
    its own step's measured peak."""

    def _graph(self, measured=None, tokens=100):
        graph = {"key": {"batch_signature": [tokens]},
                 "ops": [_op([1000], dtype="bfloat16")]}
        if measured is not None:
            graph["provenance"] = {"activation_peak_bytes": measured}
        return graph

    def test_no_measured_peak_leaves_the_walk_alone(self):
        graph = self._graph()
        assert scratch_bytes_per_token(graph) == 0.0
        assert activation_bytes_at(graph, 200) == 2 * peak_activation_bytes(graph)

    def test_the_shortfall_is_recorded_per_token(self):
        graph = self._graph(measured=peak_activation_bytes(self._graph()) + 5000,
                            tokens=100)
        assert scratch_bytes_per_token(graph) == 50.0

    def test_a_walk_that_already_matches_records_nothing(self):
        walk = peak_activation_bytes(self._graph())
        assert scratch_bytes_per_token(self._graph(measured=walk)) == 0.0

    def test_a_walk_that_over_counts_is_not_corrected_downwards(self):
        """Clamped at zero. The walk over-counting is a different fault and
        subtracting here would hide it."""
        walk = peak_activation_bytes(self._graph())
        assert scratch_bytes_per_token(self._graph(measured=walk // 2)) == 0.0

    def test_the_correction_scales_with_the_shape(self):
        """Both halves are activation memory and both are linear in tokens --
        which is what makes this worth recording at all. Fitted at 3494 tokens
        on the 27B it predicts the 4096-token warmup peak to +3.4%, against
        -35.0% for the walk alone."""
        base = self._graph()
        graph = self._graph(measured=peak_activation_bytes(base) + 5000,
                            tokens=100)
        at100 = activation_bytes_at(graph, 100)
        at200 = activation_bytes_at(graph, 200)
        assert at200 == pytest.approx(2 * at100, rel=1e-6)
