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

    t5_trace.py --model <path> --tp 2 --stage real --out tp2.json
"""

from __future__ import annotations

import argparse
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
        "module_types": sorted({type(m).__name__ for m in model.modules()}),
        "params": params,
        "buffers": buffers,
    }


def inventory(recorder: Recorder) -> dict:
    """The operator list, plus the decomposition every count must carry (7)."""
    counts: dict = {}
    for rec in recorder.ops:
        counts[rec.op] = counts.get(rec.op, 0) + 1
    return {
        "n_ops": len(recorder.ops),
        "n_distinct_ops": len(counts),
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
    ap.add_argument(
        "--tokens",
        type=int,
        default=8,
        help="trace-time hint for the token dimension; `04` D18 discipline 3 "
        "requires >= 2",
    )
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
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.tokens < 2:
        raise CaptureRefusal(
            f"--tokens {args.tokens}: a dimension whose trace-time hint is 1 is "
            "silently specialised to a constant (`04` D18 discipline 3). Trace "
            "at >= 2 and substitute afterwards."
        )
    if args.bs < 2:
        raise CaptureRefusal(
            f"--bs {args.bs}: same discipline, applied to the batch dimension."
        )

    record: dict = {
        "argv": sys.argv[1:],
        "tp": args.tp,
        "stage": args.stage,
        "tokens_hint": args.tokens,
        "skip_triton": args.skip_triton,
    }
    record["cuda_stubs"] = install_device_stubs(DeviceReadings(cu_count=args.cu_count))
    if args.stage != "build":
        record["cuda_stubs"] += install_runner_stubs()

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
            f"devices={g['param_devices']}"
        )
    err = record.get("runner_error")
    if err:
        print(f"RUNNER-REFUSED {err['type']}: {err['error']}")
        print(err["traceback"])
    f = record.get("forward")
    if f:
        if f.get("ok"):
            print(
                f"forward[{f['stage']}]: {f['n_ops']} ops, "
                f"{f['n_distinct_ops']} distinct, "
                f"attention_elided={f['attention_elided']}"
            )
        else:
            print(
                f"forward[{f['stage']}] REFUSED {f['error_type']} at {f['at']}: "
                f"{f['error']}"
            )
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
