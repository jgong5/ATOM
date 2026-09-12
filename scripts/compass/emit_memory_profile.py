"""Emit a servable memory profile per tensor-parallel width.

What the served runner consumes is `--compass-memory-model <profile.json>`;
`derived_readings` loads that file and then loads the paths *inside* it with
the same plain `open`, from the runner's own working directory. So every
reference emitted here is **absolute**: a relative sibling path works when the
emitter's cwd happens to match the server's and silently fails otherwise.

Every term is filled -- `parameters` comes from the checkpoint's own
safetensors headers at each width -- and each profile carries a `provenance`
block naming the model, the config, the checkpoint revision, the topology
probe, the environment the width deltas are only valid in, and a digest per
input. `derived_readings` ignores the block; it is there so a served
prediction can be traced back to what produced it.

The calibration also publishes `capture_reserved` at its width: the
source-only prediction of what CUDA-graph capture will reserve, replayed from
the recorded TP=1 capture request stream. Three things about it are load
bearing.

* It is **not** a budget term and `derived_readings` does not consume it. What
  leaves the KV budget is the engine's *reservation policy*, `0.2 x` the
  modelled activation peak, and that stays exactly where it was. Capture
  residency is the separate question of what capture goes on to cost.
* It needs the recorded TP=1 allocation history, which is a probe artifact a
  validator cannot hold. Computing it here, where that history lives, and
  publishing it per width is what lets a readback compare the run's measured
  pool against a number the run itself was carrying -- input-bound, and
  digested with the calibration when the run loaded the profile.
* It is source-only: the TP=1 request order transformed by which statements
  this width does not run, then replayed through the allocator's own segment
  rules. No target measurement, and no fitted constant -- in particular not
  `measured_graph_pool_bytes`, whose flat 104 MiB above TP=1 its own docstring
  marks superseded.

    emit_memory_profile.py --out DIR --checkpoint DIR --capture-history FILE
                           [--model-config FILE] [--widths 1 2 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from atom.compass.core.memory_calibration import for_model  # noqa: E402
from atom.compass.core.memory_capture import capture_stream  # noqa: E402
from atom.compass.core.memory_model import (  # noqa: E402
    built_parameter_bytes, capture_reserved_parts, weight_bytes)
from atom.compass.core.memory_topology import (  # noqa: E402
    TOPOLOGY_CONDITIONS, TOPOLOGY_PROVENANCE, compose_calibration,
    topology_delta)

MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_CONFIG = os.path.join(
    ROOT, "tests/compass/memory_records/qwen3_5_27b.config.json")
PROBE_OUT = "agent_scratch/memval/topology_out"

#: MI308X, 192 GB nominal. The one term that cannot be derived: it is the
#: *target* card's capacity, so it is stated rather than read off this box.
TOTAL = 206141652992
BUFFERS = 33554432
DTYPE_BYTES = 2
COMPILE_MODE = "inductor"

#: The registered acceptance cell. The runner derives its own warmup shape from
#: the live config, so this is not an input -- it is what the numbers below
#: were composed and evaluated at, recorded so a mismatch is visible.
CELL = {
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 32,
    "max_model_len": 262144,
    "max_num_batched_tokens": 16384,
    "block_size": 16,
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit():
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return (out.stdout or "").strip() or None
    except Exception:                                       # noqa: BLE001
        return None


def base_calibration(model: str) -> dict:
    """The TP=1 source calibration, taken from the registry rather than typed.

    `memory_calibration` already holds these constants with the run each was
    fitted on and why. Restating them here would make a second copy that can
    drift from the one the tests check, so the emitter reads the same entries
    the rest of the tree does and carries each term's own basis across as its
    provenance string.
    """
    calib = for_model(model)
    if calib is None:
        raise SystemExit(
            "no source calibration is registered for %s. The width composition "
            "starts from a measured TP=1 full-engine calibration and there is "
            "no default to fall back to." % model)
    base = dict(calib.mapping(1))
    base["provenance"] = {
        name: "S27: %s full engine at TP=%d. %s"
              % (model, term.run.tensor_parallel, term.basis)
        for name, term in calib.terms.items() if name in base
    }
    return base


def capture_prediction(trace, text_config, width: int, history: str,
                       history_sha: str) -> dict:
    """What capture will reserve at `width`, replayed from the TP=1 stream.

    Glue and nothing more: `capture_stream` transforms the recorded request
    order to this width and `capture_reserved_parts` replays it through the
    allocator's segment rules. Both are core functions with their own tests;
    what is added here is the width loop and the provenance.
    """
    stream, report = capture_stream(trace, width, text_config)
    parts = capture_reserved_parts(pool_stream=stream)
    if not parts.get("pool_reserved_derived"):
        raise SystemExit(
            "the capture replay at TP=%d did not derive its pool side, so the "
            "number would be a restatement rather than a prediction" % width)
    return {
        "total": int(parts["total"]),
        "pool_reserved": int(parts["pool_reserved"]),
        "outside_pools": int(parts["outside_pools"]),
        "pool_live_at_end": int(parts["pool_replay"]["live_at_end"]),
        "dropped_logits_in_graph": len(report["dropped_with_logits_in_graph"]),
        "unmodelled_distinct": int(report["unmodelled"]["distinct"]),
        "source_history": history,
        "source_history_sha256": history_sha,
        "provenance": (
            "S27 TP=1 capture request stream replayed at TP=%d: "
            "memory_capture.capture_stream -> "
            "memory_model.capture_reserved_parts(pool_stream=...). Source "
            "only -- no TP=%d measurement and no fitted constant. Predicts "
            "the reserved delta across the capture window, which is a "
            "different quantity from cudagraph_overhead (the engine's "
            "0.2 x peak-activation reservation policy)." % (width, width)),
    }


def _checkpoint_identity(checkpoint: str) -> dict:
    """What the weights term was read from, without hashing 55 GB of tensors.

    The index names every shard and the headers are what `weight_bytes` reads,
    so the index digest plus each shard's size pins the set that was measured.
    """
    identity = {
        "path": checkpoint,
        "revision": os.path.basename(checkpoint.rstrip("/")),
        "config_json_sha256": None,
        "index_json_sha256": None,
        "shards": [],
    }
    for name, key in (("config.json", "config_json_sha256"),
                      ("model.safetensors.index.json", "index_json_sha256")):
        path = os.path.join(checkpoint, name)
        if os.path.exists(path):
            identity[key] = sha256(path)
    for name in sorted(os.listdir(checkpoint)):
        if name.endswith(".safetensors"):
            identity["shards"].append(
                {"name": name,
                 "bytes": os.path.getsize(os.path.join(checkpoint, name))})
    return identity


def _probe_identity() -> dict:
    out = os.path.join(ROOT, PROBE_OUT)
    files = []
    if os.path.isdir(out):
        for name in sorted(os.listdir(out)):
            if name.endswith(".json"):
                files.append({"name": name,
                              "sha256": sha256(os.path.join(out, name))})
    return {"provenance": TOPOLOGY_PROVENANCE,
            "conditions": dict(TOPOLOGY_CONDITIONS),
            "readings": files}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True,
                        help="directory to write the profile set into")
    parser.add_argument("--checkpoint", required=True,
                        help="the model snapshot the weights term is read from")
    parser.add_argument("--capture-history", required=True,
                        help="the recorded TP=1 capture allocation history "
                             "(pickle) the capture term is replayed from")
    parser.add_argument("--model-config", default=DEFAULT_CONFIG,
                        help="the checkpoint's config.json")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--weights-from", choices=("built", "headers"),
                        default="built",
                        help="where the weight term comes from: ATOM's own "
                             "meta build (default, the engine's rule) or the "
                             "checkpoint headers (the approximation)")
    parser.add_argument("--replay-target",
                        help="target.json to answer AITER's architecture query "
                             "from, so --weights-from built works in a "
                             "container with no device")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.checkpoint):
        print("REFUSED: no checkpoint at %s. The weights term is read from the "
              "checkpoint's own headers and there is nothing to fall back to."
              % args.checkpoint)
        return 2
    if not os.path.exists(args.capture_history):
        print("REFUSED: no capture history at %s. The capture term is a replay "
              "of that stream and there is no constant to fall back to."
              % args.capture_history)
        return 2

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    config_path = os.path.join(out, "model_config.json")
    with open(args.model_config, encoding="utf-8") as fh:
        config = json.load(fh)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=1, sort_keys=True)

    with open(args.capture_history, "rb") as fh:
        trace = pickle.load(fh)["device_traces"][0]
    history_sha = sha256(args.capture_history)
    text_config = config.get("text_config", config)

    base = base_calibration(args.model)
    inputs = {
        "model": args.model,
        "checkpoint": _checkpoint_identity(args.checkpoint),
        "model_config_source": args.model_config,
        "model_config_sha256": sha256(config_path),
        "capture_history": args.capture_history,
        "capture_history_sha256": history_sha,
        "topology_probe": _probe_identity(),
        "memory_topology_py_sha256": sha256(
            os.path.join(ROOT, "atom/compass/core/memory_topology.py")),
        "commit": _commit(),
        "composed_at_cell": CELL,
        "composition": (
            "term(width) = S27 term at TP=1 + (standalone(width) - "
            "standalone(TP=1)); the standalone TP=1 control measured zero, so "
            "width 1 is the S27 calibration unchanged"),
        "not_inputs": [
            "no 27B TP=2 or TP=4 target record (class X27)",
            "no DEFAULT_NON_TORCH / DEFAULT_LOAD_RESIDUE entry (0.6B engine)",
            "no residual from any evaluation",
        ],
    }

    emitted = []
    for width in args.widths:
        header_params = weight_bytes(args.checkpoint, width)
        if not header_params:
            print("REFUSED: could not read checkpoint headers at %s"
                  % args.checkpoint)
            return 2
        # The headers are a table of contents, not an inventory of what the
        # engine builds: a checkpoint can ship tensors for a module the
        # configured model class never constructs, and summing the headers
        # counts them. Ask the build, and keep the headers as the fallback and
        # as a stated cross-check rather than a silent one.
        params, buffers, route = header_params, BUFFERS, "headers"
        if args.weights_from == "built":
            built = built_parameter_bytes(args.model, width,
                                          replay_target=args.replay_target)
            if built is None:
                print("WARNING: could not build %s on meta at TP=%d; the "
                      "weight term falls back to the checkpoint headers"
                      % (args.model, width))
            else:
                params, buffers = built
                route = "built"
        delta = int(header_params) - int(params)
        calibration = compose_calibration(base, width, env={})
        calibration["capture_reserved"] = {
            str(width): capture_prediction(trace, text_config, width,
                                           args.capture_history, history_sha)}
        cal_path = os.path.join(out, "calibration.tp%d.json" % width)
        with open(cal_path, "w", encoding="utf-8") as fh:
            json.dump(calibration, fh, indent=1, sort_keys=True)

        profile = {
            "total": TOTAL,
            "world_size": width,
            "parameters": int(params),
            "buffers": int(buffers),
            "model_config": config_path,
            "compile_mode": COMPILE_MODE,
            "dtype_bytes": DTYPE_BYTES,
            "calibration": cal_path,
            "provenance": dict(
                inputs,
                width=width,
                parameters=(
                    "ATOM's own meta build at TP=%d, counted once per storage "
                    "(resident_bytes); the checkpoint headers read %d B, %d B "
                    "more, which is what the engine does not construct"
                    % (width, header_params, delta)
                    if route == "built" else
                    "checkpoint safetensors headers at TP=%d, 2-D "
                    "tensors sharded and 1-D replicated; tied "
                    "embeddings counted once" % width),
                parameters_route=route,
                parameters_headers=int(header_params),
                parameters_headers_excess=delta,
                topology_delta=topology_delta(width, env={}),
                calibration_sha256=sha256(cal_path),
            ),
        }
        path = os.path.join(out, "profile.tp%d.json" % width)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=1, sort_keys=True)
        emitted.append((width, path, profile, cal_path))
        print("%s  parameters=%d  non_torch=%d  load_residue=%d  "
              "capture_reserved=%d"
              % (os.path.basename(path), profile["parameters"],
                 calibration["non_torch"][width],
                 calibration["load_residue"][width],
                 calibration["capture_reserved"][str(width)]["total"]))

    manifest = {
        "schema": "compass.memory.profile_set/1",
        "model": args.model,
        "inputs": inputs,
        "widths": {str(w): {"profile": p, "calibration": c,
                            "sha256": sha256(p),
                            "calibration_sha256": sha256(c),
                            "parameters": pr["parameters"]}
                   for w, p, pr, c in emitted},
        "how_to_serve": (
            "--compass-memory-model <profile.tpN.json> at a run whose tensor "
            "parallel size is N and which is not enforcing eager; the runner "
            "loads the referenced paths itself, which is why they are absolute"),
    }
    manifest_path = os.path.join(out, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print("MANIFEST.json  %s" % sha256(manifest_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
