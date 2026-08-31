"""Serve a fixed set of prompts and record each request's latency.

Used by `validate.py` for both halves of the comparison, so that the
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
        # Each round is one generate, so each contributes at least one prefill
        # step. Sizes are chosen to bracket what an evaluation will ask about
        # rather than to look thorough: ATOM batches several requests' prefill
        # into one step, so what lands in the table is the *batched* token
        # count, and a sweep of large prompts produces only large samples. A
        # model fitted to 1.7k-16k tokens and then asked about 500 extrapolates
        # below everything it has seen, where the intercept dominates and the
        # slope is doing no work.
        # Batch size matters as much as prompt length and is easier to forget:
        # the first version of this sweep varied only length, so it produced
        # decode steps at batch sizes 1-4 and the model was then asked about a
        # workload running 8 concurrent requests -- extrapolating outside its
        # evidence in a dimension nobody had thought to check. Coverage has to
        # bracket the evaluation in *every* dimension the model uses.
        rounds = [
            (8, 1), (16, 2), (24, 4), (32, 1), (32, 8), (48, 2), (64, 1),
            (64, 4), (64, 12), (96, 8), (128, 1), (128, 4), (128, 16),
            (192, 2), (192, 8), (256, 1), (256, 6), (256, 12), (384, 2),
            (384, 8), (512, 1), (512, 4), (768, 2), (768, 6), (1024, 1),
            (1024, 3),
        ]
        # Twice through, because Triton autotunes per shape rather than once per
        # process: the first visit to a shape pays a benchmarking cost that
        # steady-state serving never pays again. The second visit is the one
        # worth fitting, and having both lets the outlier rejection see the
        # difference rather than guess at it.
        for round_index, (length, count) in enumerate(rounds + rounds):
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
