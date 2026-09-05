"""Is the 7-second prefill real work or Triton autotuning?

Same prompts twice in one process. Autotuning is paid once per shape, so if the
second pass collapses it was compilation; if it repeats, it is the model.
"""
import sys, time, json
from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def main():
    p = FlexibleArgumentParser()
    EngineArgs.add_cli_args(p)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--out", default="/dev/null")
    a = p.parse_args()
    llm = EngineArgs.from_cli_args(a).create_engine()
    prompts = [f"Request {i}. " + " ".join(f"w{i}x{j}" for j in range(a.prompt_tokens))
               for i in range(a.num_prompts)]
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    import os
    rows = []
    passes = int(os.environ.get("PASSES", "3"))
    for label in [f"pass{i+1}" for i in range(passes)]:
        t = time.perf_counter()
        outs = llm.generate(prompts, sp)
        wall = time.perf_counter() - t
        n = len(outs)
        row = {k: sum(o.get(k, 0.0) for o in outs) / n
               for k in ("ttft", "tpot", "latency")}
        row["wall"] = wall
        rows.append(row)
        print(f"### {label:7s} wall {wall*1e3:9.1f} ms   ttft {row['ttft']*1e3:9.1f} ms"
              f"   tpot {row['tpot']*1e3:7.2f} ms"
              f"   latency {row['latency']*1e3:9.1f} ms", flush=True)
    # The first pass is compilation, not serving: on this model it is 7.1 s
    # against a warmed 320 ms. Only the warmed passes are a baseline.
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"passes": rows, "warm": rows[1:]}, fh, indent=2)


if __name__ == "__main__":
    main()
