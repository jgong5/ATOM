"""What attention costs inside a real replayed decode step.

Ablation answers it by difference, which also removes rope and the KV write and
charges any second-order effect of a smaller captured graph to attention. This
reads the kernels directly out of a profile of the real run.
"""
import glob
import json
import os
import sys

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

def main():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    args = parser.parse_args()

    llm = EngineArgs.from_cli_args(args).create_engine()
    prompts = [f"Request {i}. " + " ".join(f"w{i}x{j}" for j in range(args.prompt_tokens))
               for i in range(args.num_prompts)]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=4))
    llm.start_profile()
    llm.generate(prompts, params)
    llm.stop_profile()

    traces = sorted(glob.glob(os.path.join(args.torch_profiler_dir, "**", "*.json*"),
                              recursive=True), key=os.path.getmtime)
    print("### traces:", traces[-3:], flush=True)
    if not traces:
        sys.exit("no trace written")

    events = []
    for path in traces:
        opener = open
        if path.endswith(".gz"):
            import gzip
            opener = gzip.open
        with opener(path, "rt") as fh:
            events += json.load(fh).get("traceEvents", [])

    kernels = {}
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_user_annotation", "Kernel"):
            continue
        name = e.get("name", "")
        entry = kernels.setdefault(name, [0, 0.0])
        entry[0] += 1
        entry[1] += float(e.get("dur", 0.0))

    print("### kernels by total device time (us):", flush=True)
    for name, (n, total) in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:18]:
        print(f"###   {total:10.1f}  n={n:<6d} {total/max(n,1):8.2f}/call  {name[:70]}",
              flush=True)
    print("### PROFILE STEP DONE", flush=True)


if __name__ == "__main__":
    main()
