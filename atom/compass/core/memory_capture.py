"""Turn a recorded capture allocation history into a request program.

`memory_model.allocator_pool_bytes` replays a request/free *stream* through the
caching allocator's rules and reports what it had to map. This module produces
that stream: at the width the history was recorded at, and -- by transformation
-- at another width.

Nothing here measures anything. The one input is a `torch.cuda.memory._snapshot`
style history taken across the capture window, and every rule that reads it
says where it comes from.

**Which allocations belong to the pool.** Source-proven. Inside
`ModelRunner.capture_cudagraph`::

    4289   graph = torch.cuda.CUDAGraph()
    4290   with torch.cuda.graph(graph, self.graph_pool, stream=...):
    4293       model_output = self.model(input_ids[:num_tokens], model_positions)
    4296       outputs[:num_tokens] = model_output
    4297       if self.logits_in_graph:
    4298           graph_logits = self.model.compute_logits(outputs[:num_tokens])

4290 is the `with` statement; 4293/4296/4298 are its body. Only the body runs
under the private pool, so an allocation belongs to the pool exactly when its
`capture_cudagraph` frame stands at one of those lines. On the 27B TP=1 history
that predicate selects 7 896 allocations -- the same set an independent
attribution by segment address reached.

**Which statements the target width does not run.** Source-proven::

    # model_runner.py:4104
    is_tbo = self.config.enable_tbo and isinstance(self.model, UBatchWrapper)
    self.logits_in_graph = self.world_size == 1 and not is_tbo

Above one rank -- or under TBO at one rank -- the statement at 4298 does not
execute, so its allocations are not made. They are identified **by site, not by
size**: the frame line is the predicate. On the 27B history that is exactly 6
allocations, one per captured bucket, every one raised by `ParallelLMHead`'s
matmul, none freed inside the window; and no vocabulary-width request is made
anywhere else in the capture body, so the removal takes that family and nothing
else.

**Which widths shard.** From the checkpoint config via
`memory_model.gdn_activation_widths`, plus the vocabulary. The hidden width is
replicated.

**What is assumed, and is reported rather than hidden.**

* The execution *order* is the same at every width. It is the one input no
  config transformation produces, and a collective changes what runs between
  two allocations. `transform_report["assumptions"]` says so; a caller that
  needs it falsified needs a history at that width.
* Nothing is inserted for the collectives a TP>1 forward runs. Whether they
  allocate inside the capture is not established here.
* Requests that do not resolve to `bs x <config width> x dtype_bytes` are
  carried through unchanged, which is certainly wrong at another width. They
  are counted and returned.
"""

from typing import Mapping, Optional

from .memory_model import gdn_activation_widths

__all__ = ["capture_stream", "capture_pool_line", "CAPTURE_BODY_LINES",
           "LOGITS_IN_GRAPH_LINE", "CAPTURE_ENTER_LINE"]

#: `with torch.cuda.graph(...)` is the statement; its body is what allocates
#: into the private pool.
CAPTURE_ENTER_LINE = 4290
CAPTURE_BODY_LINES = (4293, 4296, 4298)
#: the statement `if self.logits_in_graph:` guards.
LOGITS_IN_GRAPH_LINE = 4298

_RUNNER = "model_runner.py"
_FRAME = "capture_cudagraph"


def capture_pool_line(event: Mapping) -> Optional[int]:
    """The `capture_cudagraph` frame's line for `event`, or None."""
    for frame in event.get("frames") or ():
        if (frame.get("name") == _FRAME
                and str(frame.get("filename", "")).endswith(_RUNNER)):
            return frame.get("line")
    return None


def _groups(trace):
    """Body allocations split into one group per captured bucket."""
    groups, current = [], []
    for index, event in enumerate(trace):
        if event.get("action") != "alloc":
            continue
        if capture_pool_line(event) in CAPTURE_BODY_LINES:
            current.append((index, event))
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _bucket_of(group, hidden: int, dtype_bytes: int):
    """`bs` from the group's first request, which is the embedding output."""
    first = group[0][1]["size"]
    stride = hidden * dtype_bytes
    if not stride or first % stride:
        return None
    return first // stride


def _classify(size: int, bs: int, sharded: Mapping, replicated: Mapping,
              dtype_bytes: int):
    """`(kind, width)` when the request is `bs x width x dtype_bytes`."""
    if not bs or size % dtype_bytes:
        return None
    elements = size // dtype_bytes
    if elements % bs:
        return None
    width = elements // bs
    for kind, value in sharded.items():
        if width == value:
            return kind, value
    for kind, value in replicated.items():
        if width == value:
            return kind, value
    return None


def capture_stream(trace, width: int, config: Mapping, *,
                   dtype_bytes: int = 2, logits_in_graph: Optional[bool] = None,
                   tbo: bool = False):
    """The capture pool's request program at `width`, and what it could not model.

    `trace` is one device's event list from a recorded history; `config` is the
    checkpoint's `text_config`. Returns `(stream, report)` where `stream` is an
    ordered list of `(op, key, size)` for `allocator_pool_bytes`.

    At the recorded width the transformation is the identity on every request,
    which is what makes the recorded window a control rather than a fit.
    """
    width = int(width)
    if width < 1:
        raise ValueError("width must be at least 1")
    if logits_in_graph is None:
        logits_in_graph = width == 1 and not tbo

    widths = gdn_activation_widths(config)
    hidden = int(widths["hidden"])
    sharded = {k: int(v) for k, v in widths.items() if k != "hidden"}
    sharded["vocab"] = int(config["vocab_size"])
    key = int(config["linear_num_key_heads"]) * int(config["linear_key_head_dim"])
    sharded["attn_key"] = key
    replicated = {"hidden": hidden}

    groups = _groups(trace)
    buckets = [_bucket_of(g, hidden, dtype_bytes) for g in groups]
    indexed = {}
    for group, bs in zip(groups, buckets):
        for index, event in group:
            indexed[index] = bs

    stream, unmodelled, dropped, live = [], {}, [], set()
    for index, event in enumerate(trace):
        action = event.get("action")
        if action == "alloc" and index in indexed:
            size = int(event["size"])
            if (not logits_in_graph
                    and capture_pool_line(event) == LOGITS_IN_GRAPH_LINE):
                dropped.append({"event": index, "size": size,
                                "bs": indexed[index]})
                continue
            kind = _classify(size, indexed[index], sharded, replicated,
                             dtype_bytes)
            if kind is None or (kind[0] in sharded and size % width):
                unmodelled[size] = unmodelled.get(size, 0) + 1
                scaled = size
            elif kind[0] in sharded:
                scaled = size // width
            else:
                scaled = size
            stream.append(("alloc", event["addr"], scaled))
            live.add(event["addr"])
        elif action == "free_completed" and event.get("addr") in live:
            live.discard(event["addr"])
            stream.append(("free", event["addr"], 0))

    report = {
        "width": width,
        "logits_in_graph": bool(logits_in_graph),
        "buckets": buckets,
        "requests": sum(1 for op, _, _ in stream if op == "alloc"),
        "frees": sum(1 for op, _, _ in stream if op == "free"),
        "dropped_with_logits_in_graph": dropped,
        "unmodelled": {"distinct": len(unmodelled),
                       "requests": sum(unmodelled.values())},
        "assumptions": [
            "the execution order is the one the history recorded, and is "
            "assumed unchanged at this width",
            "nothing is inserted for the collectives a TP>1 forward runs",
            "requests that do not resolve to bs x <config width> x dtype are "
            "carried unchanged",
        ],
    }
    return stream, report
