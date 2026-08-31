"""Serve a fixed set of prompts and record each request's latency.

Used by `compass_validate.py` for both halves of the comparison, so that the
real run and the simulated one differ in exactly one thing: whether the forward
pass happened. Anything else that differed would show up as model error.

Prompts are synthetic and fixed-length. Real text would make prompt length vary
with the tokenizer, and the point here is a controlled comparison, not a
realistic workload.
"""

import argparse
import json
import sys
import time

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def main() -> int:
    parser = FlexibleArgumentParser(description="ATOMCompass fixed-workload run")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sweep", action="store_true",
        help="Calibration workload: several rounds of varied prompt length and "
             "batch size, so the table holds prefill steps across a range of "
             "sizes rather than the one or two a fixed workload produces. "
             "Fitting three coefficients needs more than two samples, and a "
             "fixed workload prefills everything in a single step.",
    )
    args = parser.parse_args()

    llm = EngineArgs.from_cli_args(args).create_engine()

    # Distinct prefixes: identical prompts would share prefix-cache blocks and
    # the second request onward would skip prefill entirely, which is a real
    # ATOM behaviour but not the one being measured here.
    prompts = [
        f"Request {i}. " + " ".join(f"w{i}x{j}" for j in range(args.prompt_tokens))
        for i in range(args.num_prompts)
    ]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    if args.sweep:
        # Each round is its own generate, so each contributes at least one
        # prefill step at a different size. Lengths and batch sizes vary
        # together because that is how they vary in a deployment.
        for round_index, (length, count) in enumerate(
            [(32, 1), (128, 2), (256, 4), (512, 2), (1024, 1), (64, 8), (768, 3)]
        ):
            llm.generate(
                [f"Sweep {round_index}.{i}. "
                 + " ".join(f"s{round_index}t{i}u{j}" for j in range(length))
                 for i in range(count)],
                SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
            )
        print("sweep complete")
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"wall": 0.0, "requests": []}, fh)
        return 0

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall = time.perf_counter() - start

    requests = [
        {
            "ttft": out.get("ttft", 0.0),
            "tpot": out.get("tpot", 0.0),
            "latency": out.get("latency", 0.0),
            "num_tokens_input": out.get("num_tokens_input", 0),
            "num_tokens_output": out.get("num_tokens_output", 0),
        }
        for out in outputs
    ]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"wall": wall, "requests": requests}, fh, indent=2)
    print(f"wrote {len(requests)} requests to {args.out} (wall {wall:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
