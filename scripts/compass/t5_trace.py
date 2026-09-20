#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""P0.4 / T5: trace ATOM's model classes under `FakeTensorMode` at a logical TP
width and write the operator inventory (`12` T5, `16` P0.4, mechanism `04` D18).

Three stages, separately reportable, because they fail for different reasons:

  build   construct the module tree device-free; report the sharded geometry.
          No `ModelRunner`, no metadata, no forward.
  warmup  build a real `ModelRunner` (substitutions named in
          `fake_runner.SUBSTITUTIONS`) and trace ATOM's own `warmup_model()` --
          a 16384-token prefill batch, no KV cache.
  real    additionally allocate the KV cache through ATOM's allocator and trace
          one non-dummy decode step.

A stage that cannot complete records the refusal with its frame and exits
non-zero rather than writing a shorter inventory (principle 6).

    t5_trace.py --model <path> --tp 2 --stage real --concrete --out tp2.json

The capture refuses a symbol-free trace unless `--concrete` says that is what
was wanted: see `04` D18 discipline 2 and `02` D10.1.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time

import torch

from atom.compass.capture.fake_trace import (
    CaptureRefusal,
    DeviceReadings,
    Recorder,
    build_config,
    build_model,
    capture,
    init_single_rank_group,
    install_device_stubs,
    install_runner_stubs,
    shape_entry_census,
)


def geometry(model) -> dict:
    """What the built tree is, in the terms a width diff needs."""
    params = {
        n: [list(map(str, p.shape)), str(p.dtype)] for n, p in model.named_parameters()
    }
    buffers = {
        n: [list(map(str, b.shape)), str(b.dtype)] for n, b in model.named_buffers()
    }
    return {
        "n_modules": sum(1 for _ in model.modules()),
        "n_params": len(params),
        "n_param_elements": sum(p.numel() for p in model.parameters()),
        "n_buffers": len(buffers),
        "param_devices": sorted({str(p.device) for p in model.parameters()}),
        # T68 asks the same question of buffers that param_devices asks of
        # parameters -- `--load_dummy` and the meta wrapper act on parameters
        # only, so a buffer is the thing most likely to escape the mode.
        "buffer_devices": sorted({str(b.device) for b in model.buffers()}),
        "module_types": sorted({type(m).__name__ for m in model.modules()}),
        "params": params,
        "buffers": buffers,
    }


def inventory(recorder: Recorder) -> dict:
    """The operator list, plus the decomposition every count must carry (7)."""
    counts: dict = {}
    for rec in recorder.ops:
        counts[rec.op] = counts.get(rec.op, 0) + 1
    entries, free = shape_entry_census(recorder)
    return {
        "n_ops": len(recorder.ops),
        "n_distinct_ops": len(counts),
        # Principle 8: the one measurement that says whether this inventory is
        # a symbolic capture or the concrete T5 fallback of `02` D10.1.
        "shape_entries": entries,
        "non_numeric_shape_entries": free,
        "capture_is_symbolic": free > 0,
        "op_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "ops": [
            {"op": r.op, "in": r.in_shapes, "out": r.out_shapes, "dtypes": r.dtypes}
            for r in recorder.ops
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--stage", choices=("build", "warmup", "real"), default="build")
    ap.add_argument("--bs", type=int, default=2, help="decode rows for --stage real")
    ap.add_argument("--blocks", type=int, default=64, help="KV blocks for --stage real")
    ap.add_argument("--cu-count", type=int, default=80)
    ap.add_argument(
        "--skip-triton",
        action="store_true",
        help="record raw triton.jit launches instead of running "
        "them. DIAGNOSTIC: see TritonLaunchRecorder. The "
        "inventory it yields is an enumeration, not a cost "
        "model input.",
    )
    ap.add_argument(
        "--concrete",
        action="store_true",
        help="record a CONCRETE trace. Without this the capture refuses when "
        "no recorded shape carries a free symbol (`04` D18 discipline 2), "
        "because an empty shape_env.replacements on a symbol-free trace means "
        "nothing was checked. `02` D10.1 names the concrete result the T5 "
        "fallback; this flag is how a run asks for it on purpose.",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.bs < 2:
        raise CaptureRefusal(
            f"--bs {args.bs}: same discipline, applied to the batch dimension."
        )

    record: dict = {
        "argv": sys.argv[1:],
        "tp": args.tp,
        "stage": args.stage,
        "skip_triton": args.skip_triton,
        # B5: every headline number in this PR came from a --skip-triton run.
        # `TritonLaunchRecorder` requires runs that use it to say so, so the
        # record says it in its own field rather than only in an argv string.
        "diagnostic_inventory": args.skip_triton,
        "concrete_capture_requested": args.concrete,
    }
    readings = DeviceReadings(cu_count=args.cu_count)
    record["device_readings"] = dataclasses.asdict(readings)
    record["cuda_stubs"] = install_device_stubs(readings)
    if args.stage != "build":
        record["cuda_stubs"] += install_runner_stubs(readings)

    t0 = time.perf_counter()
    init_single_rank_group()
    record["group_init_s"] = round(time.perf_counter() - t0, 3)

    import atom

    record["atom_file"] = atom.__file__
    record["torch"] = torch.__version__
    try:
        import aiter

        record["aiter_file"] = aiter.__file__
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        record["aiter_file"] = f"UNAVAILABLE: {exc}"

    config = build_config(args.model, args.tp)
    from aiter.dist.parallel_state import get_tp_group

    group = get_tp_group()
    record["config"] = {
        "model": args.model,
        "tensor_parallel_size": config.tensor_parallel_size,
        "tp_world_size": config.tp_world_size,
        "torch_dtype": str(config.torch_dtype),
        "compilation_level": config.compilation_config.level,
        "load_dummy": config.load_dummy,
        "fake_eplb": config.fake_eplb,
        "arch": config.hf_config.architectures[0],
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "kv_cache_block_size": config.kv_cache_block_size,
        "tp_group_world_size": group.world_size,
        "tp_group_rank": group.rank_in_group,
        "tp_group_physical": getattr(
            group, "simulated_tp_physical_world_size", group.world_size
        ),
    }

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)

    t0 = time.perf_counter()
    if args.stage == "build":
        _arch, model_class, model = build_model(config, fake_mode)
        record["build_s"] = round(time.perf_counter() - t0, 3)
        record["model_class"] = f"{model_class.__module__}.{model_class.__name__}"
        record["geometry"] = geometry(model)
    else:
        from atom.compass.capture.fake_runner import SUBSTITUTIONS, CaptureModelRunner
        from scripts.compass.t5_forward import run_real, run_warmup

        record["runner_substitutions"] = SUBSTITUTIONS
        try:
            runner, model_class = CaptureModelRunner.build(config, fake_mode)
        except BaseException as exc:  # noqa: BLE001 - reported, not handled
            import traceback

            record["build_s"] = round(time.perf_counter() - t0, 3)
            record["runner_error"] = {
                "type": type(exc).__name__,
                "error": str(exc)[:2000],
                "traceback": "".join(traceback.format_tb(exc.__traceback__))[-4000:],
            }
            _emit(record, args)
            return 4
        record["build_s"] = round(time.perf_counter() - t0, 3)
        record["model_class"] = f"{model_class.__module__}.{model_class.__name__}"
        record["geometry"] = geometry(runner.model)
        if args.stage == "warmup":
            record["forward"] = run_warmup(
                runner,
                fake_mode,
                shape_env,
                capture,
                Recorder,
                inventory,
                skip_triton=args.skip_triton,
                concrete=args.concrete,
            )
        else:
            record["forward"] = run_real(
                runner,
                fake_mode,
                shape_env,
                capture,
                Recorder,
                inventory,
                bs=args.bs,
                num_blocks=args.blocks,
                skip_triton=args.skip_triton,
                concrete=args.concrete,
            )

    _emit(record, args)
    fwd = record.get("forward")
    return 0 if (fwd is None or fwd.get("ok")) else 5


def _emit(record: dict, args) -> None:
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=1, sort_keys=True)
        print("wrote", args.out)
    skip = ("geometry", "forward", "runner_substitutions")
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in skip}, indent=1, sort_keys=True
        )
    )
    g = record.get("geometry")
    if g:
        print(
            f"geometry: {g['n_modules']} modules, {g['n_params']} params, "
            f"{g['n_param_elements']} elements, {g['n_buffers']} buffers, "
            f"param_devices={g['param_devices']} "
            f"buffer_devices={g.get('buffer_devices')}"
        )
    err = record.get("runner_error")
    if err:
        print(f"RUNNER-REFUSED {err['type']}: {err['error']}")
        print(err["traceback"])
    f = record.get("forward")
    if f:
        if record.get("diagnostic_inventory"):
            print(
                "DIAGNOSTIC INVENTORY (--skip-triton): raw @triton.jit launches "
                "were recorded and NOT run, so anything downstream of a skipped "
                "kernel read uninitialised fake memory. TritonLaunchRecorder "
                "requires runs that use it to say so; this is that statement. "
                "NOT a cost model input."
            )
        if f.get("ok"):
            entries = f.get("shape_entries", 0)
            free = f.get("non_numeric_shape_entries", 0)
            print(
                f"forward[{f['stage']}]: {f['n_ops']} ops, "
                f"{f['n_distinct_ops']} distinct, "
                f"attention_elided={f['attention_elided']}"
            )
            print(
                f"  shape entries: {entries}, non-numeric (free symbols): "
                f"{free} -> capture is "
                f"{'SYMBOLIC' if free else 'CONCRETE (`02` D10.1 T5 fallback)'}"
            )
        else:
            print(
                f"forward[{f['stage']}] REFUSED {f['error_type']} at {f['at']}: "
                f"{f['error']}"
            )
            print(f"  raised at: {f.get('raised_at')}")
            print(f"  ops recorded before the refusal: {f.get('ops_before_failure')}")
            print(f["traceback"])
        t = f.get("triton")
        if t:
            print(
                f"  raw triton launches: {t['n_triton_launches']} across "
                f"{t['n_distinct_triton_kernels']} kernels (NOT dispatched, "
                f"NOT in the op inventory)"
            )
            for k in t["triton_kernels"]:
                print(f"    {k['launches']:6d}  {k['kernel']}  {k['defined_at']}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CaptureRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(3)
