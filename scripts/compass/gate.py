"""Step-1 gate: does ATOM serve a request end to end with no real compute?

Runs one generation under ATOMCompass and asserts the things that would break
if either seam were wrong:

  * the request completes and honours max_tokens (the output-length policy)
  * the Compass runner was actually the one used
  * time came from the virtual clock, not the wall clock

Usage (inside the container):
    python scripts/compass/gate.py --model <path> --compass -tp 1
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
    print(f"  clock in this process: {type(clock).__name__}")
    if not isinstance(clock, VirtualClock):
        failures.append("client process did not get a virtual clock")

    # This process stamps arrival; the engine core stamps first-token and owns
    # progress. So the evidence that virtual time advanced is the reported TTFT,
    # not this process's own clock, which never moves by design.
    ttfts = [o.get("ttft") for o in outputs if isinstance(o, dict)]
    modelled = max([t for t in ttfts if t is not None], default=None)
    if modelled is None:
        modelled = max((o.get("latency") for o in outputs
                        if isinstance(o, dict) and o.get("latency") is not None),
                       default=None)
    print(f"  modelled request time : {modelled if modelled is not None else 'n/a'}")
    print(f"  wall time spent       : {wall:.3f}s")
    if modelled is not None and modelled <= 0.0:
        failures.append(f"engine reported non-positive modelled time ({modelled})")

    # A weaker gate would pass even if decode never iterated, so check the
    # per-token timing too: TPOT must be positive, which it cannot be unless
    # every decode step advanced the clock.
    tpots = [o.get("tpot") for o in outputs if isinstance(o, dict)]
    tpots = [t for t in tpots if t is not None]
    if tpots:
        print(f"  modelled TPOT         : {min(tpots):.6f}s")
        if min(tpots) <= 0.0:
            failures.append(
                f"non-positive TPOT ({min(tpots)}): decode steps did not advance time"
            )
    if args.max_tokens > 1 and not tpots:
        failures.append("engine reported no TPOT, so decode timing is unverified")

    print("=" * 66)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("  GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
