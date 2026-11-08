# SPDX-License-Identifier: MIT
"""Fit the two carriers of a PP send, RCCL and gloo, on two GPUs of one node.

    HIP_VISIBLE_DEVICES=a,b torchrun --nproc-per-node 2 scripts/compass/pp_carriers.py OUT
    python3 scripts/compass/pp_carriers.py OUT RECORD...

`atom.compass.drive.pp_probe` probes and fits them. OUT is JSON: each
carrier's fields, the model's error on the arrival in seconds per posting order (min, p10,
p50, p90, max), every sample, and this file's and the probe's text. Given RECORDs this script
wrote, it fits their samples pooled instead of probing, and OUT names their
sha256 in place of the samples: one probe's wake moves with the run. Its
`band` is each field's p10 and p90 over the records, each fitted alone.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom.compass.drive import pp_probe
from atom.compass.drive.pp_probe import SIZES, fit, probe


def main(out: str, *records: str) -> int:
    if records:
        read = [Path(p).read_bytes() for p in records]
        samples = [s for b in read for s in json.loads(b)["samples"]]
        about = {"records": [hashlib.sha256(b).hexdigest() for b in read]}
        # Each field's p10 and p90 over the records, each fitted alone.
        alone = [fit(json.loads(b)["samples"]) for b in read]
        about["band"] = {
            c: {
                k: [float(np.percentile([a[c][k] for a in alone], q)) for q in (10, 90)]
                for k, v in alone[0][c].items()
                if not isinstance(v, list)
            }
            for c in SIZES
        }
    else:
        samples = probe()
        if not samples:
            return 0
        import torch

        about = {
            "gpus": os.environ.get("HIP_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "rccl": ".".join(map(str, torch.cuda.nccl.version())),
            "samples": samples,
        }
    carriers = fit(samples)
    print(json.dumps(carriers, indent=2))
    source = Path(__file__).read_text() + Path(pp_probe.__file__).read_text()
    record = {"carriers": carriers, **about, "source": source}
    Path(out).write_text(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
