"""Write a replay target for a width no device has run.

    python scripts/compass/derive_target.py \
        --profile profile.tp4.json --source-target target.tp1.json \
        --model Qwen/Qwen3.8-27B -tp 4 --max-model-len 262144 \
        --max-num-seqs 32 --out target.tp4.json

The block count comes from the memory model at the width asked for; the state
transfer layout, the cudagraph capture sizes and the declared card are borrowed
from the TP=1 *source* record and named in the output's `derivation.borrowed`.
No target-engine capture of the derived width is read, which is the whole
point: a record that had one would be a measurement of the deployment the
prediction is supposed to stand in for.

Prints the manifest of every file it read, so the record can be tied to the
bytes behind it.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom.compass.core.loaded_input import manifest  # noqa: E402
from atom.compass.replay.derived_target import derive_target  # noqa: E402
from atom.compass.replay.runner import TargetRecord  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True,
                    help="the memory profile for the derived width")
    ap.add_argument("--source-target", required=True,
                    help="the TP=1 source record the layout is borrowed from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-tp", "--tensor-parallel-size", type=int, required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--kv-cache-block-size", type=int, default=16)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args(argv)

    config = types.SimpleNamespace(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_block_size=args.kv_cache_block_size,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=args.enforce_eager,
    )

    read: list = []
    layout = TargetRecord.load(args.source_target)
    if layout.loaded_input:
        read.append(layout.loaded_input)
    record = derive_target(args.profile, config, layout=layout, collect=read)
    record["derivation"]["inputs"] = manifest(read)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    blocks = record["blocks"]
    print("wrote %s: TP%d, %d KV blocks, pool_entries %s"
          % (out, args.tensor_parallel_size, blocks["num_kvcache_blocks"],
             blocks["pool_entries"]))
    for row in record["derivation"]["inputs"]["inputs"]:
        print("  %-38s %s  %s" % (row["role"], row["sha256"][:12], row["path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
