"""Worker-owned configuration and capture facts, without querying a device."""

import os


def worker_snapshot(runner):
    config = runner.config
    compilation = getattr(config, "compilation_config", None)
    mode = getattr(getattr(runner, "_compass_config", None), "mode", None)
    graph_mode = getattr(compilation, "cudagraph_mode", None)
    declared = list(config.capture_sizes) if hasattr(config, "capture_sizes") else None
    fields = ("model", "tensor_parallel_size", "pipeline_parallel_size",
              "max_model_len", "max_num_seqs", "max_num_batched_tokens",
              "gpu_memory_utilization", "kv_cache_block_size", "kv_cache_dtype",
              "enforce_eager", "enable_prefix_caching")
    resolved = {key: getattr(config, key, None) for key in fields}
    resolved.update(
        compilation_level=getattr(compilation, "level", None),
        cudagraph_mode=getattr(graph_mode, "name", graph_mode),
        declared_capture_sizes=sorted(declared) if declared is not None else None,
        declared_capture_order_as_read=declared,
    )
    effective = getattr(runner, "capture_sizes", None)
    effective = [int(size) for size in effective if size] if effective is not None else None
    target = getattr(runner, "target", None)
    target_input = getattr(target, "loaded_input", None)
    native = getattr(runner, "_compass_native_capture_sizes", None) if mode == "measure" else None
    return {
        "schema": "compass.worker_runtime/1",
        "reader": {"component": "ModelRunner", "implementation": type(runner).__qualname__,
                   "pid": os.getpid(), "rank": getattr(runner, "rank", None)},
        "configuration": resolved,
        "graphs": {
            "effective_decode_buckets": sorted(effective) if effective is not None else None,
            "effective_decode_order": effective,
            "native_capture_sizes": sorted(native) if native is not None else None,
            "origin": ("borrowed_replay_target" if mode == "predict" and target is not None
                       else "native_capture" if native is not None else "not_captured"),
            "borrowed_source_capture_sizes": (sorted(target.graph.get("capture_sizes") or [])
                                               if mode == "predict" and target is not None else None),
            "borrowed_target_input": target_input.as_dict() if target_input is not None else None,
        },
    }
