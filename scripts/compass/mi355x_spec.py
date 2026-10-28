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

#: mi308x_spec.py reads the same variable for its own hosts at import.
NAME = os.environ.pop("COMPASS_MACHINE", "mi355x-gpu45")
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
#: On 005, the start is the last `INTAKE` burst's server; its 287.98 GB total
#: is `TOTAL`, the bytes rocm-smi reads on every card of 005 and -013.
if CRS005:
    MODEL = "/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro"
    DATE, COMMIT = "2026-10-28", "e4d868e7c"
    RECORD = "194520ab83874f58ca11dc7c707b9a9b2a59d2db95c96e8fa7750e077106a043"
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
#: On 005, three bursts started one after another; the record, the median of
#: both terms, ran last, after 2c1d111b and b3c71663.
if CRS005:
    INTAKE = (
        "2026-10-28",
        "e4d868e7c",
        {"fixed_s": 0.001519417945122962, "per_byte_s": 2.7286225828078513e-09},
        {
            "fixed_s": (0.0014726489628366544, 0.0016168815487359184),
            "per_byte_s": (2.4018001341410337e-09, 3.101533713093003e-09),
        },
        "f05b601cfe7a4cb3dd25e00f2c20804b162f9f3a9999cae41d65cf0d771b08e8",
    )

#: Each host's core counts as /proc/cpuinfo reads them, aiter's commit in the
#: stack its start and probe ran under (on 005 rocm/atom-dev sha256
#: 566aa3dc9d5d), and the day its cards were read with no ptl directory.
CPU, AITER, PTL_DATE = (128, 256), "9205e1059", "2026-10-18"
if CRS005:
    CPU, AITER, PTL_DATE = (118, 236), "948dd267f", "2026-10-28"

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
        "2026-10-28",
        "e4d868e7c",
        {
            "bandwidth_curve": [
                [4096, 2471999917.6953015],
                [8192, 4945851918.396276],
                [16384, 9892184763.436323],
                [32768, 19247690700.251415],
                [65536, 36885138136.18265],
                [131072, 73843420948.24696],
                [262144, 147675237955.21048],
                [524288, 283169471891.8801],
                [1048576, 561769392795.2751],
                [2097152, 1073293417417.833],
                [4194304, 1907122950230.7644],
                [8388608, 3044272193465.3496],
                [16777216, 4087521606156.239],
                [33554432, 5134029398103.868],
                [67108864, 5839143252388.048],
                [134217728, 6031685489911.714],
                [268435456, 6133849303070.915],
                [536870912, 6252795806159.282],
                [1073741824, 6451932108645.527],
                [2147483648, 6504106771819.718],
                [4294967296, 6504106771819.718],
            ],
            "bf16": 1635613360745416.0,
            "device.kernel_floor_s": [
                1.5456565022468566e-06,
                1.5445649027824403e-06,
                1.5469960689544677e-06,
            ],
            "fp4": 5649978724216232.0,
            "fp8": 3381566417063545.5,
            "interconnect.intra_node.link_latency_s": [
                8.108285069465638e-07,
                8.010325133800506e-07,
                8.201204836368562e-07,
            ],
            "link": 60866587334.964134,
        },
        "54b667a79e67c6c0fb2dccfe9c1160795e72e646254c4c8cfb0dab929469c979",
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
