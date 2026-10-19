# SPDX-License-Identifier: MIT
"""The mi355-gpu-45 MI355X machine spec, from a tokenizer swept here and one real start.

    python3 scripts/compass/mi355x_spec.py [--sweep-from DOC] [--out PATH]

The device fragment is one DeepSeek-V4-Pro start at the nightly DPA-TP8 prefill
side's arguments (`FLAGS`): the device, its capacity and the stack. One real
burst at the same arguments adds the HTTP intake terms (`INTAKE`), and
`probe_device.py`'s readings (`PEAKS`) the compute and memory derates, the
bandwidth curve, the kernel floor and the intra-node link's latency and derate.
The peaks are the datasheet's; the rest is assumed. An engine start's memory
readings are `compass measure memory`'s, not the spec's.
`--sweep-from` takes the tokenizer sweep from a document this script wrote.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mi308x_spec as base
from mi308x_spec import fragment, saved, swept

from atom.compass.spec import Fragment, merge, validate

NAME = "mi355x-gpu45"
base.NAME = NAME  # the shared fragment and sweep functions name this machine
MODEL = "/models/DeepSeek-V4-Pro"
FLAGS = "-tp 8 --enable-dp-attention --enable-tbo"

#: The total `ModelRunner` read at the second of two starts (warm compile cache).
DATE, COMMIT = "2026-10-16", "409bedc98"
TOTAL = 309220868096
#: The sha256 of the start's server log.
RECORD = "02618dade3d07bcaac510da11cf8a279a1fe3ef2fce72d41e88a76f2df7c32c6"
SUBJECT = {"model": MODEL, "width": 8, "source_root": f"commit:{COMMIT}"}
SUBJECT["record"] = f"sha256:{RECORD}"

#: `http_intake.py`'s fit over one real c=256 first-turn cc-traces burst at
#: `FLAGS`, run at the commit named; the record is the probe's stamp file. It is
#: the only burst there, so each term's band is that burst's value at both ends.
INTAKE = (
    "2026-10-18",
    "19c5ab09e",
    {"fixed_s": 0.001966482509138914, "per_byte_s": 5.288974015662166e-09},
    {
        "fixed_s": (0.001966482509138914, 0.001966482509138914),
        "per_byte_s": (5.288974015662166e-09, 5.288974015662166e-09),
    },
    "2397696f6df3a280660cbc95e7c8db0d61b77fba9986c3d388ccde75f2d5f574",
)

DEVICE = {
    "host": {"cpu": {"cores_physical": 128, "cores_logical": 256}},
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {"capacity_bytes": TOTAL},
        "software_pinned_to": {
            "rocm": "7.2.4",
            "aiter": "9205e1059",
            "rccl": "2.27.7",
            "torch": "2.10.0+rocm7.2.4.git3d3aa833",
        },
    },
}

#: AMD Instinct MI355X GPU and platform datasheets: dense peaks, the eight GPUs
#: one fabric connects, and one Infinity Fabric link's one-way rate, half its
#: 153.6 GB/s bidirectional, as a KV transfer streams one way.
DATASHEET = {
    "device": {
        "memory": {"bandwidth_bytes_per_s": 8.0e12},
        "compute": {
            "bf16_flops": 2.5166e15,
            "fp8_flops": 5.0332e15,
            "fp4_flops": 1.00663e16,
        },
    },
    "interconnect": {
        "intra_node": {"domain_devices": 8, "link_bandwidth_bytes_per_s": 7.68e10}
    },
}

#: `probe_device.py`'s readings on mi355-gpu-45 cards 0000:75:00.0 and
#: 0000:05:00.0, neither with a ptl directory, under the stack `DEVICE` pins
#: (rocm/atom-dev nightly-20261013), at the commit named; the record is its OUT.
#: FP4 is the faster of hipBLASLt and aiter. Each derate is a reading over
#: `DATASHEET`'s peak.
PEAKS = (
    "2026-10-19",
    "97a84124e",
    {
        "bf16": 1548298317647724.8,
        "fp8": 3228641949055911.0,
        "fp4": 5338794067193869.0,
        "bandwidth_curve": [
            [4096, 2617019167.9453235],
            [8192, 5210135392.980033],
            [16384, 10419077964.61944],
            [32768, 19634477709.9516],
            [65536, 38270068750.2071],
            [131072, 76495446883.72708],
            [262144, 152942734578.9677],
            [524288, 297931199356.7789],
            [1048576, 581301137385.1982],
            [2097152, 1091286499123.9933],
            [4194304, 1986422422896.641],
            [8388608, 3075660314283.9033],
            [16777216, 4114217249910.098],
            [33554432, 5163647740751.859],
            [67108864, 5860636156283.964],
            [134217728, 6043214961668.874],
            [268435456, 6149116710217.504],
            [536870912, 6318841724297.076],
            [1073741824, 6387817286185.182],
            [2147483648, 6517316964359.951],
            [4294967296, 6517316964359.951],
        ],
        "link": 61271596993.425285,
        "device.kernel_floor_s": [
            1.457617998123169e-06,
            1.4564459800720216e-06,
            1.4587820172309876e-06,
        ],
        "interconnect.intra_node.link_latency_s": [
            9.410820007324219e-07,
            9.286200165748597e-07,
            9.58860021829605e-07,
        ],
    },
    "040ae2dced8e3b5aefbff28c8e6bb5f5afe016fe373e8327fb1384407e1abef0",
)

#: No probe fills these: mi308x_spec's assumed values less what the datasheet
#: states and what `PEAKS` fills. Its inter-node link is one 400 Gb/s NIC's
#: line rate here too.
DECLARED = copy.deepcopy(base.DECLARED)
for block, groups in DATASHEET.items():
    for group, values in groups.items():
        for key in values:
            DECLARED[block][group].pop(key, None)


def device_fragments() -> list[Fragment]:
    """The start's readings, the burst's intake terms, the datasheet's peaks,
    the probed peaks and the assumed rest."""
    notes = f"one DeepSeek-V4-Pro {FLAGS} start at {COMMIT}"
    ptl = {"device": {"ptl": "disabled"}}
    datasheet = fragment(
        "datasheet", DATASHEET, "datasheet", "MI355X datasheet", date=DATE
    )
    assumed = fragment(
        "assumed", DECLARED, "assumed", "no probe fills these", date=DATE
    )
    probed, assumed = base.peaks(PEAKS, datasheet.values, assumed)
    return [
        fragment("device", DEVICE, "probed", notes, SUBJECT, DATE),
        base.intake(MODEL, *INTAKE),
        datasheet,
        assumed,
        fragment(
            "ptl", ptl, "probed", "no card has a ptl directory", date="2026-10-18"
        ),
        probed,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    out = Path(__file__).resolve().parent / "machines" / f"{NAME}.json"
    parser.add_argument("--out", type=Path, default=out)
    parser.add_argument("--sweep-from", type=Path, help="a document to reuse")
    args = parser.parse_args(argv)
    sweep = saved(args.sweep_from) if args.sweep_from else swept(MODEL)
    merged = merge([sweep, *device_fragments()])
    verdict = validate(merged.document)
    print(verdict)
    if not verdict.ok:
        return 1
    base.refuse_unbanded(base.MachineSpec.from_mapping(merged.document))
    args.out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(base.report(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
