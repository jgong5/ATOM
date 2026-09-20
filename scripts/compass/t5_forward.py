#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Stage B of P0.4 / T5: run a forward through ATOM's own `ModelRunner` under
the `04` D18 capture mechanism and record the operator inventory.

Two batches, because they cover different halves of the model and fail for
different reasons:

  warmup  ATOM's own `warmup_model()` -- a prefill batch with
          `is_dummy_run=True`. Needs no KV cache and no attention metadata,
          which is exactly why ATOM runs it before sizing the cache. It still
          contains attention: see `attention_note` on the record. What it does
          NOT contain is anything that only a real batch reaches -- the KV
          index kernel, and the six extra dispatches the decode path adds.

  real    KV cache allocated through ATOM's own `allocate_kv_cache`, then a
          decode batch with `is_dummy_run=False`, built from ATOM's own
          `Sequence` / `ScheduledBatch` -- the same construction
          `ModelRunner.dummy_execution` uses, minus `is_dummy_run`. This is the
          only one of the two that exercises ATOM's KV-cache allocator and the
          real attention-metadata build.

`skip_triton` installs `TritonLaunchRecorder`. Read its docstring before
using any inventory produced with it: it is a diagnostic that enumerates the
raw-Triton call sites a forward reaches, not a capture.
"""

from __future__ import annotations

import contextlib
import os
import traceback

import numpy as np
import torch


def _atom_root() -> str:
    """The tree under test, from the package that was actually imported."""
    import atom

    return os.path.dirname(os.path.dirname(os.path.abspath(atom.__file__)))


# The recorder sits between ATOM and torch on *every* dispatched op, so it is
# in the traceback of every refusal and never the answer to "where".
_RECORDER_FRAME = ("fake_trace.py", "__torch_dispatch__")


def _fail(stage: str, exc: BaseException) -> dict:
    """A refusal is a result: name it, keep the frame, do not widen the try.

    Two frames, because neither alone is usable. `raised_at` is `tb[-1]`,
    which for a missing fake impl is always
    `torch/_subclasses/fake_tensor.py in maybe_run_unsafe_fallback` --
    identical for every such refusal and therefore useless for telling two of
    them apart. `at` is the last frame *inside the tree under test*, which is
    the one a reader can act on: for `c10d.broadcast_` that is
    `model_runner.py:3138 in postprocess`, the `get_tp_group().broadcast()`
    call in `ModelRunner.postprocess`.

    Picking "the last non-torch frame" instead is not enough and was measured
    wrong: the dispatch path ends
    `.../torch/_library/utils.py:313 in handle_dispatch_mode`, which is torch
    but lives outside the three `torch/_subclasses|_ops|utils/_` prefixes, and
    the frame below it is aiter's `parallel_state.py:1141` -- a third-party
    call site, not an ATOM one. So the search is anchored to the ATOM tree,
    with the non-torch frame kept as a fallback for a refusal raised entirely
    outside it.
    """
    tb = traceback.extract_tb(exc.__traceback__)
    frame = tb[-1] if tb else None
    root = _atom_root()

    def _usable(f) -> bool:
        return (os.path.basename(f.filename), f.name) != _RECORDER_FRAME

    caller = next(
        (
            f
            for f in reversed(tb)
            if f.filename.startswith(root + os.sep) and _usable(f)
        ),
        None,
    )
    if caller is None:
        caller = next(
            (
                f
                for f in reversed(tb)
                if "/site-packages/torch/" not in f.filename and _usable(f)
            ),
            frame,
        )
    return {
        "ok": False,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc)[:2000],
        "at": f"{caller.filename}:{caller.lineno} in {caller.name}" if caller else "?",
        "raised_at": (
            f"{frame.filename}:{frame.lineno} in {frame.name}" if frame else "?"
        ),
        "traceback": "".join(traceback.format_tb(exc.__traceback__))[-6000:],
    }


@contextlib.contextmanager
def _triton(skip: bool):
    if not skip:
        yield None
        return
    from atom.compass.capture.fake_trace import TritonLaunchRecorder

    with TritonLaunchRecorder() as tlr:
        yield tlr


def _traced(
    stage,
    thunk,
    fake_mode,
    shape_env,
    capture_ctx,
    recorder_factory,
    inventory,
    skip_triton,
    extra,
    concrete,
):
    rec = recorder_factory()
    tlr_report = None
    try:
        with _triton(skip_triton) as tlr:
            try:
                with fake_mode, capture_ctx(shape_env, rec, concrete_ok=concrete):
                    thunk()
            finally:
                tlr_report = tlr.report() if tlr is not None else None
    except BaseException as exc:  # noqa: BLE001 - reported, not handled
        out = _fail(stage, exc)
        out["ops_before_failure"] = len(rec.ops)
        out["inventory_partial"] = inventory(rec)
        out["triton"] = tlr_report
        out["diagnostic_inventory"] = bool(skip_triton)
        out["concrete_capture_requested"] = bool(concrete)
        out.update(extra)
        return out
    out = {
        "ok": True,
        "stage": stage,
        "triton": tlr_report,
        "triton_skipped": bool(skip_triton),
        "diagnostic_inventory": bool(skip_triton),
        "concrete_capture_requested": bool(concrete),
    }
    out.update(extra)
    out.update(inventory(rec))
    return out


def run_warmup(
    runner,
    fake_mode,
    shape_env,
    capture_ctx,
    recorder_factory,
    inventory,
    skip_triton=False,
    concrete=False,
):
    """ATOM's `warmup_model()`, traced.

    Attention is NOT elided here, despite `is_dummy_run=True` -- see
    `attention_note` below, which is the measured correction to the obvious
    reading of `attention_mha.py:178`.
    """
    return _traced(
        "warmup",
        runner.warmup_model,
        fake_mode,
        shape_env,
        capture_ctx,
        recorder_factory,
        inventory,
        skip_triton,
        extra={
            "attention_elided": False,
            "traced_tokens": runner.config.max_num_batched_tokens,
            "traced_tokens_source": "config.max_num_batched_tokens, via ATOM's "
            "own warmup_model(); nothing on the command line sets it",
            "attention_note": "attention IS in this inventory despite is_dummy_run=True: "
            "aiter.unified_attention_with_output_base and "
            "aiter.linear_attention_with_output_base are registered custom "
            "ops, so FakeTensorMode answers from the registered fake impl "
            "and the Python body carrying the is_dummy_run short-circuit "
            "(attention_mha.py:178) never runs. Each is one opaque leaf; "
            "nothing below it is in the inventory.",
        },
        concrete=concrete,
    )


def _decode_batch(bs: int, block_size: int):
    """The batch `dummy_execution` builds, without `is_dummy_run`.

    ATOM's own Sequence / ScheduledBatch; the only thing constructed here is
    the block table, which a scheduler would otherwise own.
    """
    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import (
        Sequence,
        SequenceStatus,
        SequenceType,
        new_block_table,
    )

    seqs = {}
    for i in range(bs):
        seq = Sequence([0] * block_size, block_size=block_size, id=i)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = new_block_table([i])
        seqs[seq.id] = seq
    return ScheduledBatch(
        seqs=seqs,
        num_scheduled_tokens=np.ones(bs, dtype=np.int32),
        total_tokens_num=bs,
        total_tokens_num_decode=bs,
        total_seqs_num=bs,
        total_seqs_num_decode=bs,
        is_dummy_run=False,
    )


def run_real(
    runner,
    fake_mode,
    shape_env,
    capture_ctx,
    recorder_factory,
    inventory,
    bs: int,
    num_blocks: int,
    skip_triton=False,
    concrete=False,
):
    """KV cache through ATOM's allocator, then one non-dummy decode step."""
    alloc = {}
    try:
        with fake_mode:
            runner.allocate_kv_cache(num_blocks)
        alloc["num_kvcache_blocks"] = num_blocks
        alloc["kv_cache_shape"] = [
            str(s) for s in getattr(runner, "kv_cache", torch.empty(0)).shape
        ]
    except BaseException as exc:  # noqa: BLE001
        return _fail("allocate_kv_cache", exc)

    batch = _decode_batch(bs, runner.block_size)
    return _traced(
        "real",
        lambda: runner.forward(batch),
        fake_mode,
        shape_env,
        capture_ctx,
        recorder_factory,
        inventory,
        skip_triton,
        extra={
            "attention_elided": False,
            "alloc": alloc,
            "traced_tokens": bs,
            "traced_tokens_source": "--bs; a decode step is one token per row",
            "batch": {"bs": bs, "tokens": bs, "kind": "decode"},
        },
        concrete=concrete,
    )
