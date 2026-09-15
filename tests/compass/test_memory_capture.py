"""Turning a recorded capture history into a request program at another width.

The transform is the half of the graph-pool term that used to be supplied. Its
two load-bearing rules are source-proven -- which lines of `capture_cudagraph`
run inside the private pool, and which of them the target width does not run at
all -- so the tests here are about those rules holding, not about any byte
count matching a device.
"""

import pytest

from atom.compass.core.memory_capture import (
    CAPTURE_BODY_LINES, CAPTURE_ENTER_LINE, LOGITS_IN_GRAPH_LINE,
    capture_pool_line, capture_stream)

#: Enough of the 27B's `text_config` for `gdn_activation_widths` and the two
#: widths the transform adds itself.
CONFIG = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "vocab_size": 248320,
}
HIDDEN = 5120
MLP_ACT = 17408
VOCAB = 248320
DTYPE = 2


def _alloc(addr, size, line):
    return {"action": "alloc", "addr": addr, "size": size,
            "frames": [{"name": "capture_cudagraph",
                        "filename": "/x/atom/model_engine/model_runner.py",
                        "line": line}]}


def _free(addr):
    return {"action": "free_completed", "addr": addr}


def _bucket(base, bs, *, logits=True):
    """One captured bucket: the embedding output, an MLP buffer, the head."""
    trace = [_alloc(base, bs * HIDDEN * DTYPE, 4293),
             _alloc(base + 1, bs * MLP_ACT * DTYPE, 4293),
             _alloc(base + 2, bs * HIDDEN * DTYPE, 4296)]
    if logits:
        trace.append(_alloc(base + 3, bs * VOCAB * DTYPE,
                            LOGITS_IN_GRAPH_LINE))
    return trace


class TestWhichAllocationsAreInThePool:
    """Membership is the `capture_cudagraph` frame's line, not the size."""

    def test_the_with_statement_itself_is_outside_the_pool(self):
        assert CAPTURE_ENTER_LINE not in CAPTURE_BODY_LINES

    def test_an_allocation_raised_before_the_body_is_not_a_request(self):
        trace = [_alloc(1, 4096, CAPTURE_ENTER_LINE)] + _bucket(10, 8)
        stream, report = capture_stream(trace, 1, CONFIG)
        assert 1 not in [key for _, key, _ in stream]
        assert report["requests"] == 4

    def test_an_allocation_with_no_capture_frame_is_not_a_request(self):
        trace = [{"action": "alloc", "addr": 1, "size": 4096, "frames": []}]
        trace += _bucket(10, 8)
        stream, _ = capture_stream(trace, 1, CONFIG)
        assert 1 not in [key for _, key, _ in stream]

    def test_capture_pool_line_reads_only_the_runner_frame(self):
        event = {"frames": [{"name": "capture_cudagraph",
                             "filename": "/somewhere/else.py", "line": 4293}]}
        assert capture_pool_line(event) is None


class TestTheHistoryIsNotTransformedAtItsOwnWidth:
    """Width one is a control: the identity, which is what makes it one."""

    def test_every_request_is_carried_through_unchanged(self):
        trace = _bucket(10, 8) + _bucket(20, 16)
        stream, report = capture_stream(trace, 1, CONFIG)
        sizes = [size for op, _, size in stream if op == "alloc"]
        assert sizes == [event["size"] for event in trace]
        assert report["logits_in_graph"] is True
        assert report["dropped_with_logits_in_graph"] == []

    def test_the_buckets_are_read_off_the_first_request_of_each_group(self):
        # The separator is what ends a group: on a real history it is the
        # `torch.cuda.CUDAGraph()` the enter line raises between captures.
        trace = (_bucket(10, 8) + [_alloc(99, 4096, CAPTURE_ENTER_LINE)]
                 + _bucket(20, 16))
        _, report = capture_stream(trace, 1, CONFIG)
        assert report["buckets"] == [8, 16]

    def test_two_buckets_with_nothing_between_them_read_as_one(self):
        """A limitation, recorded rather than discovered later: the grouping
        is `body / not-body`, so back-to-back captures with no allocation
        between them share a bucket size. On the recorded 27B history every
        capture is separated, and the census found the 6 buckets."""
        _, report = capture_stream(_bucket(10, 8) + _bucket(20, 16), 1, CONFIG)
        assert report["buckets"] == [8]


class TestTheHeadIsDroppedBySiteNotBySize:
    """`logits_in_graph = world_size == 1 and not is_tbo` (model_runner:4104)."""

    def test_above_one_rank_the_guarded_statement_makes_no_request(self):
        trace = _bucket(10, 8)
        stream, report = capture_stream(trace, 2, CONFIG)
        assert [event["size"] for event in trace][3] not in [
            size for _, _, size in stream]
        assert len(report["dropped_with_logits_in_graph"]) == 1
        assert report["dropped_with_logits_in_graph"][0]["bs"] == 8

    def test_a_vocabulary_sized_request_from_another_line_survives(self):
        """Size is not the predicate: the same bytes from 4293 are kept."""
        trace = [_alloc(10, 8 * HIDDEN * DTYPE, 4293),
                 _alloc(11, 8 * VOCAB * DTYPE, 4293)]
        stream, report = capture_stream(trace, 2, CONFIG)
        assert report["dropped_with_logits_in_graph"] == []
        assert 8 * VOCAB * DTYPE // 2 in [size for _, _, size in stream]

    def test_one_rank_under_tbo_also_does_not_capture_the_head(self):
        stream, report = capture_stream(_bucket(10, 8), 1, CONFIG, tbo=True)
        assert report["logits_in_graph"] is False
        assert len(report["dropped_with_logits_in_graph"]) == 1

    def test_the_predicate_can_be_overridden_explicitly(self):
        _, report = capture_stream(_bucket(10, 8), 4, CONFIG,
                                   logits_in_graph=True)
        assert report["logits_in_graph"] is True
        assert report["dropped_with_logits_in_graph"] == []


class TestSharding:
    """Config widths shard; the hidden width does not."""

    def test_a_sharded_width_is_divided(self):
        trace = [_alloc(10, 8 * HIDDEN * DTYPE, 4293),
                 _alloc(11, 8 * MLP_ACT * DTYPE, 4293)]
        stream, _ = capture_stream(trace, 4, CONFIG)
        assert dict((key, size) for _, key, size in stream)[11] == \
            8 * MLP_ACT * DTYPE // 4

    def test_the_hidden_width_is_replicated(self):
        trace = [_alloc(10, 8 * HIDDEN * DTYPE, 4293)]
        stream, _ = capture_stream(trace, 4, CONFIG)
        assert stream[0][2] == 8 * HIDDEN * DTYPE

    def test_a_request_that_matches_no_config_width_is_counted_not_guessed(self):
        trace = [_alloc(10, 8 * HIDDEN * DTYPE, 4293),
                 _alloc(11, 8 * 777 * DTYPE, 4293)]
        stream, report = capture_stream(trace, 2, CONFIG)
        assert report["unmodelled"]["distinct"] == 1
        assert stream[1][2] == 8 * 777 * DTYPE, "carried unchanged, not scaled"


class TestFrees:
    def test_a_free_of_a_request_in_the_stream_is_a_free(self):
        trace = _bucket(10, 8) + [_free(11)]
        stream, report = capture_stream(trace, 1, CONFIG)
        assert ("free", 11, 0) in stream
        assert report["frees"] == 1

    def test_a_free_of_something_outside_the_pool_is_not_in_the_stream(self):
        trace = _bucket(10, 8) + [_free(999)]
        _, report = capture_stream(trace, 1, CONFIG)
        assert report["frees"] == 0

    def test_the_dropped_heads_free_is_dropped_with_it(self):
        """The `logits_in_graph` case: the allocation never happened, so the
        free is not a free of anything and must not reach the replay."""
        trace = _bucket(10, 8) + [_free(13)]
        stream, report = capture_stream(trace, 2, CONFIG)
        assert report["frees"] == 0
        assert 13 not in [key for _, key, _ in stream]


class TestTheReportSaysWhatItAssumed:
    def test_the_assumptions_are_returned_rather_than_left_in_a_docstring(self):
        _, report = capture_stream(_bucket(10, 8), 2, CONFIG)
        text = " ".join(report["assumptions"])
        assert "execution order" in text
        assert "collectives" in text

    def test_a_width_below_one_is_refused(self):
        with pytest.raises(ValueError):
            capture_stream(_bucket(10, 8), 0, CONFIG)
