"""Deriving the activation term by walking the graph.

Not how much memory the operators touch -- how much is live at once. That needs
to know which tensor is which, which shapes alone cannot say, so the trace
records which operator produced each input.
"""

import json
import struct

import pytest

from atom.compass.core.memory_model import (
    CAPTURE_FIXED_PINNED, DEFAULT_NON_TORCH, DEFAULT_PERSISTENT,
    DEFAULT_POOL_FLOOR, capture_pinned_bytes,
    activation_bytes_at, activation_curve, graph_pool_bytes,
    load_residue_bytes, measured_graph_pool_bytes, modelled_readings,
    non_torch_bytes, peak_activation_bytes, scratch_bytes_per_token,
    liveness_is_recorded, liveness_instrumentation, traced_shape,
    LIVENESS_INSTRUMENTATION, UNVERSIONED_LIVENESS, UnfoundedActivation,
    allocator_block_bytes, allocator_segment_bytes, allocator_charged_bytes,
    capture_reserved_parts, ALLOCATOR_SMALL_SIZE, allocator_pool_bytes,
    ALLOCATOR_SMALL_BUFFER, ALLOCATOR_LARGE_BUFFER,
    GDN_ACTIVATION_INSTANTS, activation_instant_bytes, gdn_activation_widths,
    UnfoundedPrediction, derived_readings, weight_bytes)


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
                 "ops": [_op([1000], dtype="bfloat16", dies_at=0)]}
        assert activation_bytes_at(graph, 200) == 2 * peak_activation_bytes(graph)

    def test_a_graph_that_names_no_shape_is_taken_as_it_stands(self):
        graph = {"ops": [_op([1000], dtype="bfloat16", dies_at=0)]}
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
                 "ops": [_op([1000], dtype="bfloat16", dies_at=0)]}
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


class TestWhatTheGraphDoesNotSay:
    """Two questions the walk was answering by guessing, now asked out loud."""

    def test_a_derivation_records_no_deaths_and_says_so(self):
        """The 27B's meta-derived prefill graphs, in miniature.

        Every graph derived before 2026-09-11 reached the walk with no
        `dies_at` -- not because meta cannot observe a death, which it can, but
        because only the capture path stamped what the tracer had watched.
        `_deaths` still returns a map, falling back to last-read, and the walk
        still returns a number. On `s27prefhead.tp1` that number is
        570 425 344 B against a measured 2 956 984 320 B: 19.3% of the term,
        all of it from 64 `aten::empty.memory_format` allocations with no
        recorded death. The figure is not wrong so much as unfounded, and the
        caller has to be able to tell.
        """
        derived = {"ops": [dict(_op([1024]), dies_at=[-1]),
                           dict(_op([1024], inputs_from=[0]), dies_at=[-1])]}
        captured = {"ops": [_op([1024], dies_at=1), _op([1024], inputs_from=[0])]}

        assert liveness_is_recorded(derived) is False
        assert liveness_is_recorded(captured) is True
        # And the walk answers anyway, which is the point.
        assert peak_activation_bytes(derived) > 0

    def test_a_graph_with_no_dies_at_field_at_all_is_not_recorded_liveness(self):
        graph = {"ops": [{"name": "aten::x", "output_shapes": [[1024]],
                          "dtypes": ["float32"], "output_aliases": [None],
                          "inputs_from": []}]}
        assert liveness_is_recorded(graph) is False

    def test_a_graph_that_does_not_name_its_producer_is_the_old_one(self):
        """Absent is not unknown.

        Every artifact on disk predates the field and every one of them came
        off the same producer, so an unstamped graph is version 1 and reading
        it as "cannot say" would let an old template pass a check it never
        met. The fields it carries look tidy either way -- that is exactly why
        the question has to be asked of the provenance and not of the fields.
        """
        assert liveness_instrumentation({"ops": []}) == UNVERSIONED_LIVENESS
        assert liveness_instrumentation(
            {"provenance": {"source": "derivation"}}) == UNVERSIONED_LIVENESS
        assert liveness_instrumentation(
            {"provenance": {"liveness_instrumentation": 2}}) == 2
        # A graph written by a producer newer than this reader still reports
        # its own number; the reader's job is to say what it was given.
        assert liveness_instrumentation(
            {"provenance": {"liveness_instrumentation": 7}}) == 7
        assert LIVENESS_INSTRUMENTATION > UNVERSIONED_LIVENESS

    def test_the_shape_is_queries_and_history_not_a_token_total(self):
        """`s27prefhead` and `s27prefdeep` have identical keys.

        Both are `batch_signature [16384]`; one starts cold and the other
        carries 98 304 cached tokens. Anything matching on the sum takes
        whichever it is handed. The spec the derivation was asked for is in
        the provenance, and that is where the difference lives.
        """
        head = {"key": {"batch_signature": [16384]},
                "provenance": {"batch_spec": {"query_lens": [16384],
                                              "context_lens": [16384]}}}
        deep = {"key": {"batch_signature": [16384]},
                "provenance": {"batch_spec": {"query_lens": [16384],
                                              "context_lens": [114688]}}}
        assert head["key"] == deep["key"]
        assert traced_shape(head) == ((16384,), (16384,))
        assert traced_shape(deep) == ((16384,), (114688,))

    def test_a_capture_says_its_shape_in_its_own_words(self):
        """`provenance.shape`, which is what a device trace writes."""
        graph = {"key": {"batch_signature": [3494]},
                 "provenance": {"shape": {"num_scheduled_tokens": [3494],
                                          "context_lens": [3494]}}}
        assert traced_shape(graph) == ((3494,), (3494,))

    def test_an_unlabelled_graph_reports_no_history_rather_than_zero(self):
        """`()` is "unknown", and the caller must not read it as "cold"."""
        assert traced_shape({"key": {"batch_signature": [512, 512]}}) == (
            (512, 512), ())

    def test_a_budget_is_refused_rather_than_built_on_a_guess(self):
        """`activation_bytes_at` is the budget-facing entry point.

        Refusing is the whole point: a caller sizing a configuration nobody has
        run has nothing to fall back on, so a quiet 19% understatement becomes
        a pool sized too large and an engine that cannot start. A measured peak
        is enough on its own -- `scratch_bytes_per_token` carries whatever the
        walk missed -- so only a graph with neither is refused.
        """
        guessed = {"key": {"batch_signature": [100]},
                   "ops": [dict(_op([1000], dtype="bfloat16"), dies_at=[-1])]}
        with pytest.raises(UnfoundedActivation):
            activation_bytes_at(guessed, 200)

        measured = dict(guessed,
                        provenance={"activation_peak_bytes": 4000})
        assert activation_bytes_at(measured, 100) == 4000

        walked = {"key": {"batch_signature": [100]},
                  "ops": [_op([1000], dtype="bfloat16", dies_at=0)]}
        assert activation_bytes_at(walked, 200) == 4000


#: A calibration that could found a prediction: every term the model would
#: otherwise default, and a provenance class for each saying it was fitted at
#: the source configuration rather than read off the target.
_FOUNDED_CALIBRATION = {
    "persistent": 252339712,
    "non_torch": {"1": 1157627904},
    "load_residue": {"1": 14924832},
    "provenance": {"persistent": "S27", "non_torch": "S27",
                   "load_residue": "S27"},
}

#: The 27B at TP=1 on an MI308X, as a profile naming its two side files.
_FOUNDED_PROFILE = {
    "total": 206141652992, "world_size": 1, "parameters": 54713457120,
    "buffers": 33554432, "graph": "graph.json",
    "calibration": "calibration.json",
}


def _founded(profile=None, calibration=None, graph=None):
    """A profile, a loader for its side files, and whatever was overridden."""
    walked = {"key": {"batch_signature": [100], "topology": [["tp", 1]]},
              "ops": [_op([1000], dtype="bfloat16", dies_at=0)]}
    files = {"graph.json": graph if graph is not None else walked,
             "calibration.json": dict(_FOUNDED_CALIBRATION, **(calibration or {}))}
    return dict(_FOUNDED_PROFILE, **(profile or {})), files.__getitem__


class TestAPredictionThatCannotBeMadeIsRefused:
    """`derived_readings` fails closed, term by term.

    The failure this guards against is not a wrong number, it is a *mislabelled*
    one. Every refusal here was, until now, a silent substitution: a missing
    activation peak became zero, an absent `total` became whatever card the run
    happened to land on, a missing calibration became constants fitted on a
    0.6B, and any exception on the way became device sizing. Each produced a
    budget that a reader could not distinguish from a forecast. So the test for
    every one of them is that it raises, and that the message names the term --
    a refusal nobody can act on is only a different kind of dead end.
    """

    def test_a_graph_traced_at_another_width_is_not_stretched_to_this_one(self):
        """The activation peak is the one term that shards.

        Everything else in the budget is either flat in width or has its own
        per-width table; the peak is walked from a graph, and a walk cannot be
        re-sharded after the fact. Handing a TP=1 graph to a TP=4 prediction
        would have produced a number four ranks wide in a budget one rank wide,
        with nothing in the readings to show for it. This is the gate a TP>1
        prediction currently stops at, and stopping is correct until a graph is
        traced at that width.
        """
        profile, load = _founded({"world_size": 4})
        with pytest.raises(UnfoundedPrediction) as refusal:
            derived_readings(profile, warmup_tokens=200, load=load)
        assert "TP=1" in str(refusal.value) and "TP=4" in str(refusal.value)

    def test_a_graph_that_does_not_say_its_width_is_not_assumed_narrow(self):
        """Silence is not width one -- that guess is the whole failure."""
        profile, load = _founded(
            graph={"key": {"batch_signature": [100]},
                   "ops": [_op([1000], dtype="bfloat16", dies_at=0)]})
        with pytest.raises(UnfoundedPrediction) as refusal:
            derived_readings(profile, warmup_tokens=200, load=load)
        assert "topology" in str(refusal.value)

    def test_a_founded_profile_derives_all_five(self):
        profile, load = _founded()
        readings, activation = derived_readings(
            profile, warmup_tokens=200, load=load)
        assert sorted(readings) == ["cudagraph_overhead", "free", "non_torch",
                                    "peak_torch", "total"]
        assert readings["total"] == 206141652992
        assert activation == 4000
        assert readings["peak_torch"] == (
            54713457120 + 33554432 + 14924832 + 252339712 + 4000)

    def test_no_capacity_is_not_this_cards_capacity(self):
        """The one term that cannot be derived, and so has to be given.

        Reading it from `mem_get_info` sized the prediction to whichever box
        the modelling run was launched on, which is the one input a prediction
        for another machine must not take from here.
        """
        profile, load = _founded({"total": 0})
        with pytest.raises(UnfoundedPrediction, match="total"):
            derived_readings(profile, warmup_tokens=200, load=load)

    def test_no_graph_is_not_a_zero_activation(self):
        """Zero was the old default and it is not a small error.

        The activation term is the single largest derived quantity in the
        budget -- 2.96 GB at the 27B source config. Defaulting it to zero frees
        that much for KV, and the engine dies at steady state rather than at
        start-up, where the cause would have been obvious.
        """
        profile, load = _founded({"graph": None})
        with pytest.raises(UnfoundedPrediction, match="activation"):
            derived_readings(profile, warmup_tokens=200, load=load)

    def test_a_graph_with_nothing_to_walk_reaches_the_caller(self):
        """`UnfoundedActivation` is a refusal too, and must not be swallowed."""
        graph = {"key": {"batch_signature": [100], "topology": [["tp", 1]]},
                 "ops": [dict(_op([1000], dtype="bfloat16"), dies_at=[-1])]}
        profile, load = _founded(graph=graph)
        with pytest.raises(UnfoundedActivation):
            derived_readings(profile, warmup_tokens=200, load=load)
        assert issubclass(UnfoundedActivation, UnfoundedPrediction)

    def test_no_warmup_shape_is_refused(self):
        """A peak is a peak *at a shape*; without one there is nothing to scale."""
        profile, load = _founded()
        with pytest.raises(UnfoundedPrediction, match="max_num_batched_tokens"):
            derived_readings(profile, warmup_tokens=0, load=load)

    def test_no_parameters_is_refused(self):
        profile, load = _founded({"parameters": 0})
        with pytest.raises(UnfoundedPrediction, match="parameter"):
            derived_readings(profile, warmup_tokens=200, load=load)

    def test_no_calibration_is_not_the_built_in_defaults(self):
        """The defaults are a fallback wearing the clothes of a derivation.

        `DEFAULT_PERSISTENT` and `DEFAULT_NON_TORCH` were fitted on a different
        model at a different width. Standing in for a missing calibration is
        the same substitution the device fallback made, in a smaller place and
        harder to see.
        """
        profile, load = _founded({"calibration": None})
        with pytest.raises(UnfoundedPrediction, match="calibration"):
            derived_readings(profile, warmup_tokens=200, load=load)

    @pytest.mark.parametrize("term", ["persistent", "non_torch",
                                      "load_residue"])
    def test_a_calibration_missing_a_term_names_it(self, term):
        profile, load = _founded(calibration={term: None})
        with pytest.raises(UnfoundedPrediction, match=term):
            derived_readings(profile, warmup_tokens=200, load=load)

    def test_a_calibration_without_provenance_is_refused(self):
        """Numbers without an origin cannot be checked against the rule below."""
        profile, load = _founded(calibration={"provenance": {}})
        with pytest.raises(UnfoundedPrediction, match="provenance"):
            derived_readings(profile, warmup_tokens=200, load=load)

    @pytest.mark.parametrize("term", ["persistent", "non_torch",
                                      "load_residue"])
    def test_provenance_has_to_cover_every_term_it_supplies(self, term):
        provenance = dict(_FOUNDED_CALIBRATION["provenance"])
        provenance[term] = ""
        profile, load = _founded(calibration={"provenance": provenance})
        with pytest.raises(UnfoundedPrediction, match=term):
            derived_readings(profile, warmup_tokens=200, load=load)

    def test_a_term_fitted_on_the_target_is_refused_outright(self):
        """Class X is a measurement of the configuration being predicted.

        Not a weaker prediction -- not one at all. The number would agree with
        the target because it *is* the target, and the agreement would be
        reported as a successful forecast. There is no use for the result, so
        it is refused rather than flagged.
        """
        provenance = dict(_FOUNDED_CALIBRATION["provenance"],
                          non_torch="X27")
        profile, load = _founded(calibration={"provenance": provenance})
        with pytest.raises(UnfoundedPrediction, match="non_torch"):
            derived_readings(profile, warmup_tokens=200, load=load)

    def test_the_source_class_is_not_caught_by_the_target_rule(self):
        """S27 is an authorised source-configuration fit and stays allowed."""
        profile, load = _founded()
        readings, _ = derived_readings(profile, warmup_tokens=200, load=load)
        assert readings["non_torch"] == 1157627904


class TestWhatCapturePins:
    """The pinned half of capture, as a mechanism instead of a width constant.

    `measured_graph_pool_bytes` predicts the reserved delta with a line fitted
    at one width and a constant above it. This one states what is in the pool:
    a fixed residue, plus the LM head when the runner captures it.
    """

    #: The 27B's TP=1 source record: ladder, vocabulary, and the allocated
    #: delta capture reported (`tests/compass/memory_records/27b.tp1.memory
    #: .json`, `qwen3_5_27b.config.json`).
    LADDER = (1, 2, 4, 8, 16, 32)
    VOCAB = 248320
    RECORDED_ALLOCATED = 110981120

    def test_the_source_record_is_reproduced_to_the_byte(self):
        assert capture_pinned_bytes(self.LADDER,
                                    vocab_size=self.VOCAB) == self.RECORDED_ALLOCATED

    def test_the_head_leaves_the_graph_above_width_one(self):
        """`logits_in_graph = world_size == 1 and not is_tbo`. Above width one
        nothing in the pinned set scales with the ladder, which is the whole
        content of the `DEFAULT_POOL_SHARDED` constant."""
        for width in (2, 4, 8):
            assert (capture_pinned_bytes(self.LADDER, world_size=width)
                    == capture_pinned_bytes((1,), world_size=width)
                    == CAPTURE_FIXED_PINNED)

    def test_tbo_at_width_one_drops_the_head_as_well(self):
        """The predicate is not the width, which is why it is not read as one.
        A run that read `world_size == 1` would over-read by the whole ladder
        term here."""
        assert capture_pinned_bytes(self.LADDER, vocab_size=self.VOCAB,
                                    tbo=True) == CAPTURE_FIXED_PINNED

    def test_a_captured_head_with_no_vocabulary_refuses(self):
        with pytest.raises(UnfoundedPrediction):
            capture_pinned_bytes(self.LADDER)

    def test_the_ladder_enters_as_tokens_not_as_batches(self):
        """Buckets are `bs x max_q_len`; a spec-decode run captures q>1, and
        the head is sized in tokens."""
        assert (capture_pinned_bytes(self.LADDER, vocab_size=self.VOCAB, q_len=4)
                - CAPTURE_FIXED_PINNED
                == 4 * (self.RECORDED_ALLOCATED - CAPTURE_FIXED_PINNED))

    def test_capturing_nothing_pins_nothing(self):
        assert capture_pinned_bytes((), vocab_size=self.VOCAB) == 0
        assert capture_pinned_bytes(self.LADDER, vocab_size=self.VOCAB,
                                    enforce_eager=True) == 0

    def test_the_residue_is_calibratable_without_touching_the_mechanism(self):
        pinned = capture_pinned_bytes(
            self.LADDER, vocab_size=self.VOCAB,
            calibration={"graph_pool": {"fixed_pinned": 1000}})
        assert pinned == 1000 + self.VOCAB * 2 * sum(self.LADDER)


class TestTheReservedSideFollowsTheAllocatorsOwnRules:
    """The reserved delta is not the allocated one rounded.

    Every figure checked here comes from the S27 TP=1 pool-id probe
    (`agent_scratch/memval/pool_probe/att2_artifact.json`) and the constants
    come from `c10/core/AllocatorConfig.h`. Neither was fitted.
    """

    def test_a_request_under_the_block_size_still_costs_a_block(self):
        assert allocator_block_bytes(8) == 512
        assert allocator_block_bytes(513) == 1024

    def test_the_three_segment_classes_are_the_allocators_not_ours(self):
        assert allocator_segment_bytes(8) == ALLOCATOR_SMALL_BUFFER
        assert allocator_segment_bytes(1_048_576) == ALLOCATOR_SMALL_BUFFER
        # over kSmallSize but under kMinLargeAlloc: a whole large segment
        assert allocator_segment_bytes(1_048_577) == ALLOCATOR_LARGE_BUFFER
        assert allocator_segment_bytes(9_000_000) == ALLOCATOR_LARGE_BUFFER
        # at or over kMinLargeAlloc: its own segment, rounded to 2 MiB
        assert allocator_segment_bytes(10_485_760) == 10_485_760
        assert allocator_segment_bytes(15_892_480) == 16_777_216

    def test_the_fixed_residue_maps_the_bytes_that_were_observed(self):
        """76 MiB exactly, plus a whole 2 MiB segment for 1 KiB of scalars."""
        parts = capture_reserved_parts(46_137_344)
        assert parts["outside_pools"] == 81_788_928
        assert parts["outside_pools_derived"] is True

    def test_the_allocated_constant_is_not_its_reserved_cost(self):
        parts = capture_reserved_parts(46_137_344)
        assert parts["outside_pools"] != CAPTURE_FIXED_PINNED
        assert parts["outside_pools"] - CAPTURE_FIXED_PINNED == 2_096_128

    def test_the_window_total_is_reproduced_only_with_the_measured_pool(self):
        """The observed reserved delta, once the pool is supplied.

        Supplied, not predicted: the pool half is the high-water mark of the
        captured forward, and `capture_reserved_parts` says so by taking it as
        an argument.
        """
        parts = capture_reserved_parts(46_137_344)
        assert parts["total"] == 127_926_272
        assert parts["pool_reserved_derived"] is False

    def test_the_pinned_set_does_not_predict_the_pool(self):
        """Why the pool half is an input: the same rule over what capture
        pins reads 82% high against the pool that was measured."""
        pinned = (15_892_480, 7_946_240, 3_973_120,
                  1_986_560, 993_280, 496_640)
        naive = sum(allocator_segment_bytes(size) for size in pinned)
        assert naive == 83_886_080
        assert naive - 46_137_344 == 37_748_736


QWEN3_27B = {"hidden_size": 5120, "intermediate_size": 17408,
             "linear_num_key_heads": 16, "linear_key_head_dim": 128,
             "linear_num_value_heads": 48, "linear_value_head_dim": 128}


class TestAnInstantBelongsToTheProgramItWasWitnessedIn:
    """The history is an Inductor run and the walk is not, so they don't mix."""

    def test_the_widths_come_out_of_the_config_and_nowhere_else(self):
        widths = gdn_activation_widths(QWEN3_27B)
        # q, k, v, z at 2 x 2048 + 2 x 6144, then b and a at one per value head.
        assert widths["in_proj_qkvzba"] == 16_480
        assert widths["mlp_gate_up"] == 34_816
        assert widths["mlp_act"] == 17_408
        assert widths["attn_value"] == 6_144
        assert widths["hidden"] == 5_120

    def test_each_instant_names_the_program_it_came_from(self):
        assert (GDN_ACTIVATION_INSTANTS["linear_attn"]["compile_mode"]
                == "inductor")
        assert GDN_ACTIVATION_INSTANTS["mlp_down"]["compile_mode"] == "eager"

    def test_the_two_instants_hold_the_mixes_the_witnesses_recorded(self):
        widths = gdn_activation_widths(QWEN3_27B)
        for name, expected in (("linear_attn", (74_848, 15_360)),
                               ("mlp_down", (52_224, 30_720))):
            instant = GDN_ACTIVATION_INSTANTS[name]
            sharded = sum(widths[k] for k in instant["sharded"])
            replicated = sum(widths[k] for k in instant["replicated"])
            assert (sharded, replicated) == expected

    def test_a_mode_with_no_witness_refuses_rather_than_borrowing_one(self):
        with pytest.raises(UnfoundedActivation, match="not evidence about"):
            activation_instant_bytes(QWEN3_27B, 16_384, 1,
                                     compile_mode="cudagraph")

    def test_the_compiled_mode_lands_on_the_measured_gate_bar_the_replay_gap(self):
        best = activation_instant_bytes(QWEN3_27B, 16_384, 1,
                                        compile_mode="inductor")
        assert best["instant"] == "linear_attn"
        assert best["bytes"] == 2_955_935_744
        # The one measurement of the source config, and the gap it leaves.
        assert 2_956_984_320 - best["bytes"] == 1_048_576
        assert best["is_candidate"] is True

    def test_the_compiled_mode_reports_what_it_cannot_count_above_one_rank(self):
        one = activation_instant_bytes(QWEN3_27B, 16_384, 1,
                                       compile_mode="inductor")
        assert one["uncounted"] == ()
        for width, expected in ((2, 1_729_626_112), (4, 1_116_471_296)):
            best = activation_instant_bytes(QWEN3_27B, 16_384, width,
                                            compile_mode="inductor")
            assert best["bytes"] == expected
            assert len(best["uncounted"]) == 1
            assert "unwitnessed" in best["uncounted"][0]

    def test_the_eager_mode_carries_the_collective_destination_it_witnessed(self):
        assert activation_instant_bytes(
            QWEN3_27B, 16_384, 1, compile_mode="eager")["bytes"] == 2_717_908_992
        for width, expected in ((2, 2_030_043_136), (4, 1_602_224_128)):
            best = activation_instant_bytes(QWEN3_27B, 16_384, width,
                                            compile_mode="eager")
            assert best["bytes"] == expected
            assert best["uncounted"] == ()
            assert best["replicated"] - 30_720 == 5_120

    def test_the_two_modes_disagree_at_tp1_in_both_directions(self):
        widths = gdn_activation_widths(QWEN3_27B)
        held = 16_384 * 2 * (widths["in_proj_qkvzba"] + widths["attn_value"])
        reused = 16_384 * 2 * 3 * widths["hidden"]
        assert held == 741_343_232          # compiled run holds these longer
        assert reused == 503_316_480        # ... and reuses these
        eager = activation_instant_bytes(QWEN3_27B, 16_384, 1,
                                         compile_mode="eager")["bytes"]
        compiled = activation_instant_bytes(QWEN3_27B, 16_384, 1,
                                            compile_mode="inductor")["bytes"]
        assert eager + held - reused == compiled

    def test_it_scales_with_tokens_and_dtype_and_nothing_else(self):
        one = activation_instant_bytes(QWEN3_27B, 1, 1,
                                       compile_mode="inductor")["bytes"]
        assert activation_instant_bytes(
            QWEN3_27B, 16_384, 1, compile_mode="inductor")["bytes"] == 16_384 * one
        assert activation_instant_bytes(
            QWEN3_27B, 16_384, 1, compile_mode="inductor",
            dtype_bytes=4)["bytes"] == 2 * 16_384 * one

    def test_a_width_that_does_not_divide_the_shard_refuses(self):
        with pytest.raises(UnfoundedActivation, match="not the split"):
            activation_instant_bytes(QWEN3_27B, 16_384, 3,
                                     compile_mode="inductor")


class TestAPredictionCanReachTheInstantWithoutAGraph:
    """The second route into the activation term, and what it costs to use it.

    The walk needs a graph traced at the target width, which is why every TP>1
    prediction stopped at that gate. The config-derived instant does not: its
    widths come from the checkpoint and the instant itself is witnessed once,
    at the source. What it needs instead is the *program* -- the same instant
    is not the same number under Inductor as under eager -- so the profile has
    to name the compile mode, and a profile that does not is refused rather
    than defaulted onto whichever program the witness happened to be.
    """

    @staticmethod
    def _with_config(profile=None):
        """`_founded`, plus a checkpoint config the loader can hand back."""
        base, load = _founded(profile)
        def loader(path):
            if path == "config.json":
                return {"text_config": QWEN3_27B}
            return load(path)
        return base, loader

    def test_a_config_and_a_mode_reach_a_width_no_graph_was_traced_at(self):
        """The gate a TP=4 prediction used to stop at, passed on config alone."""
        profile, load = self._with_config(
            {"model_config": "config.json", "compile_mode": "inductor",
             "graph": None, "world_size": 4})
        _, activation = derived_readings(
            profile, warmup_tokens=16_384, load=load)
        assert activation == 1_116_471_296

    def test_a_config_without_a_mode_is_refused_and_not_defaulted(self):
        """Choosing the program is the caller's, and it is not a small choice.

        The two witnessed instants differ by 238 MB at the source config. A
        default would pick one, and the prediction would carry no sign of which.
        """
        profile, load = self._with_config(
            {"model_config": "config.json", "graph": None, "world_size": 4})
        with pytest.raises(UnfoundedPrediction, match="compile_mode"):
            derived_readings(profile, warmup_tokens=16_384, load=load)

    def test_the_config_is_taken_over_the_graph_when_both_are_named(self):
        """One term, one derivation: the walk is not a second opinion."""
        profile, load = self._with_config(
            {"model_config": "config.json", "compile_mode": "eager"})
        _, activation = derived_readings(profile, warmup_tokens=200, load=load)
        assert activation == activation_instant_bytes(
            QWEN3_27B, 200, 1, compile_mode="eager")["bytes"]
        assert activation != 4000        # what the graph would have walked

    def test_a_refusal_from_the_instant_reaches_the_caller_intact(self):
        """`UnfoundedActivation` is not caught and re-dressed on the way out."""
        profile, load = self._with_config(
            {"model_config": "config.json", "compile_mode": "cudagraph",
             "graph": None})
        with pytest.raises(UnfoundedActivation, match="not evidence about"):
            derived_readings(profile, warmup_tokens=16_384, load=load)

    def test_naming_neither_still_refuses(self):
        profile, load = self._with_config({"graph": None})
        with pytest.raises(UnfoundedPrediction, match="activation"):
            derived_readings(profile, warmup_tokens=16_384, load=load)

    def test_a_profile_with_no_config_walks_the_graph_exactly_as_before(self):
        profile, load = self._with_config()
        _, activation = derived_readings(profile, warmup_tokens=200, load=load)
        assert activation == 4000


class TestWhatTheAllocatorChargesIsNotWhatWasAsked:
    """The 1 MiB the source prediction was missing, as a rule rather than a fit.

    `allocated_bytes` charges the block the allocator hands out. When a fresh
    segment's leftover is too small to be worth splitting off, it is not split
    off -- it stays inside that block, and the charge is the whole segment. A
    snapshot shows nothing free, because nothing is free.

    The threshold was read off the shipped binary's behaviour, since the rule
    lives in a `.cpp` the wheel does not ship: in the S27 TP=1 warmup snapshot
    the largest retained remainder is 1 048 576 and the smallest split-off tail
    is 1 114 112, so the boundary is exactly `kSmallSize`, witnessed from both
    sides.
    """

    def test_the_source_in_proj_request_retains_exactly_one_mebibyte(self):
        """16 384 x 16 480 x 2 B, and the gap the diagnostic could not close.

        The warmup history witnesses this allocation directly: a segment of
        541 065 216 B mapped, the 540 016 640 B request served at the same
        address, no allocation ever at the tail address, and the segment freed
        as one unit.
        """
        charged = allocator_charged_bytes(16_384 * 16_480 * 2)
        assert charged["requested"] == 540_016_640
        assert charged["block"] == 540_016_640
        assert charged["segment"] == 541_065_216
        assert charged["retained"] == 1_048_576
        assert charged["charged"] == 541_065_216

    def test_a_remainder_one_block_over_the_threshold_is_split_off(self):
        """The other side of the boundary, so the rule is not one-sided."""
        charged = allocator_charged_bytes(541_065_216 - 1_048_576 - 512)
        assert charged["retained"] == 0
        assert charged["split_off"] > ALLOCATOR_SMALL_SIZE
        assert charged["charged"] == charged["block"]

    @pytest.mark.parametrize("width,tail", [(1, 1_048_576), (2, 524_288),
                                            (4, 0)])
    def test_it_does_not_simply_halve_with_width(self, width, tail):
        """The reason this is a rule and not a per-width constant.

        Halving the shard halves the request but not the rounding: at TP=4 the
        remainder lands at 1 310 720, *over* `kSmallSize`, so it is split off
        and there is nothing retained at all. A term fitted at TP=1 and halved
        would have claimed 262 144 B here.
        """
        assert 16_480 % width == 0
        charged = allocator_charged_bytes(16_384 * (16_480 // width) * 2)
        assert charged["retained"] == tail

    def test_every_other_tensor_in_the_instant_rounds_exactly(self):
        """Which is why one allocation accounts for the whole source gap."""
        for elements in (34_816, 17_408, 6_144, 5_120):
            charged = allocator_charged_bytes(16_384 * elements * 2)
            assert charged["retained"] == 0
            assert charged["charged"] == charged["requested"]

    def test_a_request_served_from_existing_space_claims_only_the_block(self):
        """Whose segment it lands in is history, not a property of the request."""
        charged = allocator_charged_bytes(540_016_640, fresh_segment=False)
        assert charged["segment"] is None
        assert charged["charged"] == 540_016_640
        assert charged["retained"] == 0

    def test_a_small_request_is_charged_its_block_and_not_its_buffer(self):
        """The small pool splits down to `kMinBlockSize`, so 2 B costs 512 B."""
        charged = allocator_charged_bytes(2)
        assert charged["block"] == 512
        assert charged["segment"] == ALLOCATOR_SMALL_BUFFER
        assert charged["charged"] == 512


class TestThePoolIsReplayedFromRequestsNotFromWhatSurvived:
    """The capture pool's reserved half, derived rather than supplied.

    `capture_pinned_bytes` explains what capture *pins*. The same rule over
    those pinned blocks reads 82% high against the pool that was measured,
    because the pool's segments are the high-water mark of the whole captured
    forward and nearly all of it is freed before the window closes. What the
    allocator responds to is the *order* of requests and frees, so that is what
    `allocator_pool_bytes` is handed.
    """

    def test_a_free_lets_the_next_request_reuse_the_segment(self):
        """Two requests, one segment -- because the first died first."""
        stream = [("alloc", "a", 300_000), ("free", "a", 0),
                  ("alloc", "b", 300_000)]
        replay = allocator_pool_bytes(stream)
        assert replay["segments_mapped"] == 1
        assert replay["reserved"] == ALLOCATOR_SMALL_BUFFER

    def test_the_same_requests_in_a_different_order_cost_different_bytes(self):
        """Order, not the set of sizes, is what the allocator responds to.

        Three 900 000 B requests. Two blocks fit in one 2 MiB buffer and the
        third does not, so held together they cost two segments; freed as they
        go they cost one. Same three sizes either way.
        """
        together = allocator_pool_bytes([("alloc", "a", 900_000),
                                         ("alloc", "b", 900_000),
                                         ("alloc", "c", 900_000)])
        serial = allocator_pool_bytes([("alloc", "a", 900_000),
                                       ("free", "a", 0),
                                       ("alloc", "b", 900_000),
                                       ("free", "b", 0),
                                       ("alloc", "c", 900_000)])
        assert together["segments_mapped"] == 2
        assert together["reserved"] == 2 * ALLOCATOR_SMALL_BUFFER
        assert serial["segments_mapped"] == 1
        assert serial["reserved"] == ALLOCATOR_SMALL_BUFFER
    def test_two_small_requests_share_one_small_segment(self):
        """A 2 MiB buffer holds both, because the remainder splits off."""
        stream = [("alloc", "a", 900_000), ("alloc", "b", 900_000)]
        replay = allocator_pool_bytes(stream)
        assert replay["segments_mapped"] == 1
        assert replay["reserved"] == ALLOCATOR_SMALL_BUFFER

    def test_a_segment_is_reported_with_the_request_that_forced_it(self):
        """Which request mapped a segment is not which tensor ends up in it.

        This is the whole reason the pool cannot be predicted from what capture
        pins: in the S27 window every pool segment was forced by a transient,
        and the survivors landed in the space those transients left.
        """
        stream = [("alloc", "big", 1_054_720), ("free", "big", 0),
                  ("alloc", "survivor", 2_000_000)]
        replay = allocator_pool_bytes(stream)
        assert replay["forced_by"] == [1_054_720]
        assert replay["segments"] == [ALLOCATOR_LARGE_BUFFER]
        assert replay["live_at_end"] == allocator_block_bytes(2_000_000)

    def test_a_freed_large_block_does_not_serve_a_small_request(self):
        """The two size pools are separate, so 20 MiB free buys nothing."""
        stream = [("alloc", "big", 1_054_720), ("free", "big", 0),
                  ("alloc", "small", 496_640)]
        replay = allocator_pool_bytes(stream)
        assert replay["segments"] == [ALLOCATOR_LARGE_BUFFER,
                                      ALLOCATOR_SMALL_BUFFER]
        assert replay["forced_by"] == [1_054_720, 496_640]
    def test_a_free_with_no_allocation_is_ignored(self):
        """A transformation may drop a source branch and leave its free.

        `logits_in_graph` is the case in hand: above one rank the statement at
        `model_runner.py:4298` does not run, so a stream derived from a TP=1
        history has frees whose allocations are gone.
        """
        replay = allocator_pool_bytes([("free", "never-allocated", 0)])
        assert replay["reserved"] == 0
        assert replay["segments_mapped"] == 0

    def test_an_unknown_op_is_refused_rather_than_skipped(self):
        with pytest.raises(ValueError):
            allocator_pool_bytes([("realloc", "a", 1)])

    def test_the_parts_say_whether_the_pool_was_derived(self):
        supplied = capture_reserved_parts(46_137_344)
        assert supplied["pool_reserved_derived"] is False
        derived = capture_reserved_parts(pool_stream=[("alloc", "a", 300_000)])
        assert derived["pool_reserved_derived"] is True
        assert derived["pool_reserved"] == ALLOCATOR_SMALL_BUFFER
        assert derived["total"] == derived["outside_pools"] + ALLOCATOR_SMALL_BUFFER

    def test_neither_a_figure_nor_a_program_is_refused(self):
        with pytest.raises(UnfoundedPrediction):
            capture_reserved_parts()

    def test_both_a_figure_and_a_program_is_refused(self):
        """A recorded figure and a replay are two different claims."""
        with pytest.raises(UnfoundedPrediction):
            capture_reserved_parts(46_137_344,
                                   pool_stream=[("alloc", "a", 300_000)])


class TestAProfileHasToBeAboutTheRunThatLoadsIt:
    """Two ways a founded profile can still describe a different deployment.

    Both were silent. The width one matters most: every width-dependent term is
    keyed off the profile's own `world_size`, and `_at_width` answers a width
    it has no entry for from the widest one below it -- so a TP=2 profile
    handed to a TP=4 run produced a complete, confident, wrongly-sized budget
    with nothing in the readings to show which width it was for.
    """

    @staticmethod
    def _with_config(profile=None):
        base, load = _founded(profile)
        def loader(path):
            if path == "config.json":
                return {"text_config": QWEN3_27B}
            return load(path)
        return base, loader

    def test_a_profile_for_another_width_is_refused_not_stretched(self):
        profile, load = self._with_config(
            {"model_config": "config.json", "compile_mode": "inductor",
             "graph": None, "world_size": 2})
        with pytest.raises(UnfoundedPrediction) as refusal:
            derived_readings(profile, warmup_tokens=16_384, load=load,
                             world_size=4)
        assert "TP=2" in str(refusal.value) and "TP=4" in str(refusal.value)

    def test_the_matching_width_passes_through(self):
        profile, load = self._with_config(
            {"model_config": "config.json", "compile_mode": "inductor",
             "graph": None, "world_size": 4})
        _, activation = derived_readings(profile, warmup_tokens=16_384,
                                         load=load, world_size=4)
        assert activation == 1_116_471_296

    def test_a_caller_that_does_not_know_the_width_is_not_forced_to_guess(self):
        """Omitting it is how a device-free caller says it has no deployment."""
        profile, load = self._with_config(
            {"model_config": "config.json", "compile_mode": "inductor",
             "graph": None, "world_size": 4})
        readings, _ = derived_readings(profile, warmup_tokens=16_384, load=load)
        assert readings["peak_torch"] > 0

    def test_an_inductor_profile_is_refused_for_an_eager_run(self):
        """`enforce_eager` zeroes the graph pool and moves the instant.

        Left unchecked the two halves of one prediction describe two programs:
        a compiled activation peak beside an eager graph-pool term.
        """
        profile, load = self._with_config(
            {"model_config": "config.json", "compile_mode": "inductor",
             "graph": None, "world_size": 4})
        with pytest.raises(UnfoundedPrediction, match="enforce_eager"):
            derived_readings(profile, warmup_tokens=16_384, load=load,
                             enforce_eager=True)

    def test_an_eager_profile_is_what_an_eager_run_wants(self):
        profile, load = self._with_config(
            {"model_config": "config.json", "compile_mode": "eager",
             "graph": None, "world_size": 4})
        readings, _ = derived_readings(profile, warmup_tokens=16_384,
                                       load=load, enforce_eager=True)
        assert readings["cudagraph_overhead"] == 0


class TestACalibrationIsOnlyValidInItsOwnEnvironment:
    """The pools move between two terms and neither number looks wrong.

    Under expandable segments or a raw collective input pool, 2 GiB per rank
    crosses from `load_residue` into `non_torch`. A prediction built on the
    other environment is still complete, still provenanced and still exactly as
    confident, which is why the environment is checked rather than documented.
    Opt-in by data: a calibration that states no conditions is untouched.
    """

    @staticmethod
    def _with_conditions(monkeypatch, conditions, env):
        base, load = _founded(calibration={"conditions": conditions})
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        return base, load

    def test_the_measured_environment_passes(self, monkeypatch):
        profile, load = self._with_conditions(
            monkeypatch, {"PYTORCH_HIP_ALLOC_CONF": None},
            {"PYTORCH_HIP_ALLOC_CONF": None})
        readings, _ = derived_readings(profile, warmup_tokens=200, load=load)
        assert readings["peak_torch"] > 0

    def test_a_different_allocator_environment_is_refused(self, monkeypatch):
        profile, load = self._with_conditions(
            monkeypatch, {"PYTORCH_HIP_ALLOC_CONF": None},
            {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"})
        with pytest.raises(UnfoundedPrediction) as refusal:
            derived_readings(profile, warmup_tokens=200, load=load)
        assert "PYTORCH_HIP_ALLOC_CONF" in str(refusal.value)
        assert "2 GiB per rank" in str(refusal.value)

    def test_a_calibration_that_states_nothing_is_unaffected(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
        profile, load = _founded()
        readings, _ = derived_readings(profile, warmup_tokens=200, load=load)
        assert readings["peak_torch"] > 0

    def test_the_conditions_are_read_from_the_composed_block_too(self, monkeypatch):
        """`compose_calibration` nests them under `topology_delta`."""
        monkeypatch.setenv("AITER_CUSTOM_AR_RAW_INPUT_POOL", "1")
        profile, load = _founded(calibration={
            "topology_delta": {"conditions": {"AITER_CUSTOM_AR_RAW_INPUT_POOL": None}}})
        with pytest.raises(UnfoundedPrediction,
                           match="AITER_CUSTOM_AR_RAW_INPUT_POOL"):
            derived_readings(profile, warmup_tokens=200, load=load)
