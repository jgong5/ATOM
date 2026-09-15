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
        compilation_cache_dir=getattr(compilation, "local_cache_dir", None),
        cudagraph_mode=getattr(graph_mode, "name", graph_mode),
        declared_capture_sizes=sorted(declared) if declared is not None else None,
        declared_capture_order_as_read=declared,
    )
    effective = getattr(runner, "capture_sizes", None)
    effective = [int(size) for size in effective if size] if effective is not None else None
    target = getattr(runner, "target", None)
    target_input = getattr(target, "loaded_input", None)
    native = getattr(runner, "_compass_native_capture_sizes", None) if mode == "measure" else None
    result = {
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

    if mode == "measure" and os.environ.get("COMPASS_NATIVE_COVERAGE") == "1":
        result["native_attention"] = native_attention_snapshot(runner)
    return result


def native_attention_snapshot(runner):
    """Read instantiated modules and bound views through the existing scope ABI."""
    from atom.compass.core.cost.families.attention_scope import read_resolved
    from atom.model_ops.fla_ops import chunk_o, l2norm
    from atom.utils import envs, forward_context

    def qualname(value):
        cls = value if isinstance(value, type) else type(value)
        return cls.__module__ + "." + cls.__qualname__

    def tensor(value):
        return {"shape": list(value.shape), "stride": list(value.stride()),
                "dtype": str(value.dtype), "element_size": value.element_size(),
                "is_contiguous": value.is_contiguous()}

    fields = ("activation", "head_k_dim", "head_v_dim", "hidden_size", "key_dim",
              "num_k_heads", "num_v_heads", "value_dim", "layer_num",
              "kv_cache_dtype", "sliding_window")
    layers, views, state_slots = {}, {}, set()
    bound = forward_context._forward_kv_cache_context.kv_cache_data
    for label, layer in runner.config.compilation_config.static_forward_context.items():
        if not hasattr(layer, "attn_backend") or not hasattr(layer, "impl"):
            continue
        layers[label] = {"class": qualname(layer), "attn_backend": qualname(layer.attn_backend),
                         "impl": qualname(layer.impl),
                         "impl_attrs": {k: getattr(layer.impl, k) for k in fields
                                        if hasattr(layer.impl, k)}}
        number = layers[label]["impl_attrs"].get("layer_num", getattr(layer, "layer_num", None))
        view = bound[f"layer_{number}"]
        views[f"layer_{number}"] = {"k": tensor(view.k_cache), "v": tensor(view.v_cache)}
        if qualname(layer) == "atom.model_ops.base_attention.LinearAttention":
            state_slots.add(int(view.k_cache.shape[0]))
    if len(state_slots) != 1:
        raise ValueError("native GDN bound state-slot counts disagree")
    names = ("ATOM_USE_UNIFIED_ATTN", "ATOM_FORCE_ATTN_TRITON", "ATOM_V4_BACKEND",
             "ATOM_V4_BACKEND_LAYERS", "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST")
    record = {"layers": layers, "kv_views": views,
              "atom_envs": {name: getattr(envs, name) for name in names},
              "config": {"kv_cache_dtype": runner.config.kv_cache_dtype,
                         "kv_cache_block_size": runner.config.kv_cache_block_size},
              "caches_summary": {"state_slots": next(iter(state_slots)),
                                 "pool_shapes": {"kv_cache": [
                                     list(runner.kv_cache.shape), str(runner.kv_cache.dtype)]}}}
    return {"record": record, "declaration": read_resolved(record).as_dict(),
            "generated_cache_paths": {key: os.environ[key] for key in (
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR")},
            "body_flags": {"FLA_GDN_FIX_BT": bool(chunk_o.FLA_GDN_FIX_BT),
                           "USE_DEFAULT_FLA_NORM": int(l2norm.USE_DEFAULT_FLA_NORM)}}


def native_batch_allocation(batch):
    """Copy the scheduler's assignment before native forward consumes it."""
    ops = getattr(batch, "state_maintenance_ops", None)
    return {"source": "ScheduledBatch",
            "block_tables": [list(map(int, row)) for row in batch.block_tables],
            "cached_tokens": list(map(int, batch.num_cached_tokens)),
            "state_rows": list(batch.state_rows),
            "state_slots": list(batch.state_slots_committed),
            "state_fork_srcs": list(batch.state_fork_srcs),
            "num_prefill_seqs": int(batch.total_seqs_num_prefill),
            "state_maintenance": None if ops is None else {
                "relocations": list(ops.relocations),
                "checkpoint_stores": len(ops.checkpoint_stores),
                "checkpoint_restores": len(ops.checkpoint_restores)}}
