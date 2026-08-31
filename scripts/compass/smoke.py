"""Minimal generation smoke test: does this model run at all on this box?

Deliberately does *not* touch Compass, so a failure here is an environment or
model-support problem, not a Compass one.
"""

import argparse
import sys

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def main() -> int:
    parser = FlexibleArgumentParser(description="ATOM generation smoke test")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    llm = EngineArgs.from_cli_args(args).create_engine()
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    for out in outputs:
        print("OUTPUT:", out)
    print("SMOKE: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
