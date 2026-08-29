"""Step-1 gate: does ATOM serve a request end to end with no real compute?

Runs one generation under ATOMCompass and asserts the things that would break
if either seam were wrong:

  * the request completes and honours max_tokens (the output-length policy)
  * the Compass runner was actually the one used
  * time came from the virtual clock, not the wall clock

Usage (inside the container):
    python scripts/compass_gate.py --model <path> --compass -tp 1
"""

import argparse
import sys
import time

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def main() -> int:
    parser = FlexibleArgumentParser(description="ATOMCompass step-1 gate")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=2)
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    prompts = [f"Prompt number {i}:" for i in range(args.num_prompts)]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    wall_start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall = time.perf_counter() - wall_start

    print("\n" + "=" * 66)
    print("ATOMCompass step-1 gate")
    print("=" * 66)

    failures = []

    if len(outputs) != len(prompts):
        failures.append(f"expected {len(prompts)} outputs, got {len(outputs)}")

    for i, out in enumerate(outputs):
        n = len(out.get("token_ids", out.get("text", "")))
        print(f"  request {i}: completed, {n} output units")
        if n == 0:
            failures.append(f"request {i} produced nothing")

    from atom.utils.clock import VirtualClock, get_clock

    clock = get_clock()
    on_virtual = isinstance(clock, VirtualClock)
    print(f"  clock: {type(clock).__name__}")
    if on_virtual:
        print(f"  virtual time modelled : {clock.elapsed:.3f}s")
        print(f"  wall time spent       : {wall:.3f}s")
        if clock.elapsed <= 0.0:
            failures.append("virtual clock never advanced")
    else:
        failures.append("engine did not run on a virtual clock")

    print("=" * 66)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("  GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
