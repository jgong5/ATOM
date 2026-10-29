# SPDX-License-Identifier: MIT
"""An MI355X machine spec, from the tokenizers swept here and one real start.

    python3 scripts/compass/mi355x_spec.py [--model M ...] [--sweep-from DOC] [--out PATH]

Builds mi355-gpu-45's spec, or with COMPASS_MACHINE=mi355x-crs005
crsuse2-m2m-v3-005's, which -013 shares: the same host CPU, cards, VBIOS and
driver, and no ptl directory.
The device fragment is one DeepSeek-V4-Pro start at the nightly DPA-TP8 prefill
side's arguments (`FLAGS`): the device, its capacity and the stack. One real
burst at the same arguments adds the HTTP intake terms (`INTAKE`), and
`probe_device.py`'s readings (`PEAKS`) the compute and memory derates, the
bandwidth curve, the kernel floor and the intra-node link's latency and derate.
The peaks are the datasheet's; the rest is assumed. An engine start's memory
readings are `compass measure memory`'s, not the spec's.
`--model` and `--sweep-from` work as `mi308x_spec.py`'s do; with neither the
script sweeps DeepSeek-V4-Pro's tokenizer.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The machines this script has tables for. mi308x_spec.py reads the same
#: variable for its own hosts at import.
MACHINES = ("mi355x-gpu45", "mi355x-crs005")
NAME = os.environ.pop("COMPASS_MACHINE", MACHINES[0])
if NAME not in MACHINES:
    raise SystemExit(f"COMPASS_MACHINE={NAME}: this script builds only {MACHINES}")
CRS005 = NAME == "mi355x-crs005"

import mi308x_spec as base
from mi308x_spec import fragment, saved, swept

from atom.compass.spec import Fragment, merge, validate

base.NAME = NAME  # the shared fragment and sweep functions name this machine
MODEL = "/models/DeepSeek-V4-Pro"
FLAGS = "-tp 8 --enable-dp-attention --enable-tbo"

#: The total `ModelRunner` read at the second of two starts (warm compile cache).
DATE, COMMIT = "2026-10-16", "409bedc98"
TOTAL = 309220868096
#: The sha256 of the start's server log.
RECORD = "02618dade3d07bcaac510da11cf8a279a1fe3ef2fce72d41e88a76f2df7c32c6"
#: On 005, the start is the `INTAKE` record's server; its 287.98 GB total
#: is `TOTAL`, the bytes rocm-smi reads on every card of 005 and -013.
if CRS005:
    MODEL = "/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro"
    DATE, COMMIT = "2026-10-29", "f6e1fa77d"
    RECORD = "622cbece7a489b55bf75879131b4456d21af34daa31e8294ba63625f6a354dfb"
SUBJECT = {"model": MODEL, "width": 8, "source_root": f"commit:{COMMIT}"}
SUBJECT["record"] = f"sha256:{RECORD}"

#: `http_intake.py`'s fit over one real c=256 first-turn cc-traces burst at
#: `FLAGS`, run at the commit named; the record is the probe's stamp file. The
#: band is each term's p10 and p90 over six bursts, each fitted alone: the
#: record and five more at 10a0609f7 on 2026-10-19, stamp files sha256
#: 4e304c7a, 246e816a, 69148498, 0b987d21 and 59cac66f.
INTAKE = (
    "2026-10-18",
    "19c5ab09e",
    {"fixed_s": 0.001966482509138914, "per_byte_s": 5.288974015662166e-09},
    {
        "fixed_s": (0.0019362225299832747, 0.0023826020769630186),
        "per_byte_s": (3.617351613835432e-09, 5.241243893276638e-09),
    },
    "2397696f6df3a280660cbc95e7c8db0d61b77fba9986c3d388ccde75f2d5f574",
)
#: On 005, three bursts started one after another with no gate on either
#: node and no other deployment on 005: the record, 133dde61 and ae9b8a49; the
#: record is the median of the fixed term, the larger.
if CRS005:
    INTAKE = (
        "2026-10-29",
        "f6e1fa77d",
        {"fixed_s": 0.0020912355743506537, "per_byte_s": 4.19943165426115e-09},
        {
            "fixed_s": (0.0020310947130036056, 0.002139447088691459),
            "per_byte_s": (3.9301788350529524e-09, 4.159128855060515e-09),
        },
        "fc9dadb1074821306edad43745d303c9de754feb03e9efcb195885b7282663cb",
    )

#: Each host's core counts as /proc/cpuinfo reads them, aiter's commit in the
#: stack its start and probe ran under (on 005 rocm/atom-dev sha256
#: 4d44267b1d5a, nightly 20261026), and the day its cards were read with no ptl directory.
CPU, AITER, PTL_DATE = (128, 256), "9205e1059", "2026-10-18"
if CRS005:
    CPU, AITER, PTL_DATE = (118, 236), "41d4d8d7e", "2026-10-29"

DEVICE = {
    "host": {"cpu": {"cores_physical": CPU[0], "cores_logical": CPU[1]}},
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {"capacity_bytes": TOTAL},
        "software_pinned_to": {
            "rocm": "7.2.4",
            "aiter": AITER,
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
#: On 005, cards 0003:00:03.0 and 0003:00:04.0.
if CRS005:
    PEAKS = (
        "2026-10-29",
        "f6e1fa77d",
        {
            "bandwidth_curve": [
                [4096, 2448213119.8574038],
                [8192, 4901932899.645926],
                [16384, 9799057786.896889],
                [32768, 19216086138.60756],
                [65536, 36700073375.20446],
                [131072, 73396013852.64185],
                [262144, 146726346018.3001],
                [524288, 281614855936.05884],
                [1048576, 559324208073.2725],
                [2097152, 1069975261609.8036],
                [4194304, 1900125139330.0735],
                [8388608, 3036138588634.988],
                [16777216, 4078284050135.474],
                [33554432, 5133415851480.742],
                [67108864, 5842638294512.771],
                [134217728, 6020018863349.566],
                [268435456, 6119695251585.363],
                [536870912, 6257719852339.275],
                [1073741824, 6371910488551.815],
                [2147483648, 6463389077030.495],
                [4294967296, 6556731501062.837],
            ],
            "bf16": 1637694846431171.0,
            "device.kernel_floor_s": [
                1.5658164620399475e-06,
                1.5642129778861997e-06,
                1.5676080465316773e-06,
            ],
            "fp4": 5634127727361962.0,
            "fp8": 3384221638626554.5,
            "interconnect.intra_node.link_latency_s": [
                8.638092577457427e-07,
                8.557889819145204e-07,
                8.71113008260727e-07,
            ],
            "link": 60842596092.87844,
        },
        "ba2c62a63f3adfdee2541e81b81aa03b153ec8c563f92aa54fbcab7a0a573cf1",
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
        fragment("ptl", ptl, "probed", "no card has a ptl directory", date=PTL_DATE),
        probed,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    out = Path(__file__).resolve().parent / "machines" / f"{NAME}.json"
    parser.add_argument("--out", type=Path, default=out)
    parser.add_argument("--model", action="append", help="repeats; default " + MODEL)
    parser.add_argument("--sweep-from", type=Path, help="a document to reuse")
    args = parser.parse_args(argv)
    sweeps = [saved(args.sweep_from)] if args.sweep_from else []
    if args.model or not sweeps:
        sweeps += swept(args.model or [MODEL])
    merged = merge([*sweeps, *device_fragments()])
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
