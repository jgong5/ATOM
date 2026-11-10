# SPDX-License-Identifier: MIT
"""An MI355X machine spec, from one real start and the probes named below.

    python3 scripts/compass/mi355x_spec.py [--out PATH]

Builds mi355-gpu-45's spec, or with COMPASS_MACHINE=mi355x-crs005
crsuse2-m2m-v3-005's, which -013 shares: the same cards, VBIOS and driver, and
no ptl directory. With COMPASS_MACHINE=mi355x-gpu40 it builds mi355-gpu-40's,
under rocm/atom-dev nightly 20261026's stack, the first that starts
DeepSeek-V4.1-Flash, with its aiter as `aiter_c90842add0.sh` leaves it.
The device fragment is one DeepSeek-V4-Pro start at the nightly DPA-TP8 prefill
side's arguments (`FLAGS`): the device, its capacity and the stack; on gpu-40,
whose stack that start did not run, the stack is `STACK`'s.
`probe_device.py`'s readings (`PEAKS`) add the compute and memory derates, the
bandwidth curve, the kernel floor and the intra-node link's latency and derate.
On crsuse2-m2m-v3-005, the network file of its rails into -013 adds the
inter-node link (`NETWORK`). The peaks are the datasheet's; the rest is
assumed. An engine start's memory readings are `compass measure memory`'s, not
the spec's.
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
MACHINES = ("mi355x-gpu45", "mi355x-crs005", "mi355x-gpu40")
NAME = os.environ.pop("COMPASS_MACHINE", MACHINES[0])
if NAME not in MACHINES:
    raise SystemExit(f"COMPASS_MACHINE={NAME}: this script builds only {MACHINES}")
CRS005, GPU40 = NAME == "mi355x-crs005", NAME == "mi355x-gpu40"

import mi308x_spec as base
from mi308x_spec import fragment

from atom.compass.spec import Fragment, merge, validate

base.NAME = NAME  # the shared fragment functions name this machine
MODEL = "/models/DeepSeek-V4-Pro"
FLAGS = "-tp 8 --enable-dp-attention --enable-tbo"

#: The total `ModelRunner` read at the second of two starts (warm compile cache).
DATE, COMMIT = "2026-10-16", "409bedc98"
TOTAL = 309220868096
#: The sha256 of the start's server log.
RECORD = "02618dade3d07bcaac510da11cf8a279a1fe3ef2fce72d41e88a76f2df7c32c6"
#: On 005, the start served a c=256 cc-traces burst; its 287.98 GB total
#: is `TOTAL`, the bytes rocm-smi reads on every card of 005 and -013.
if CRS005:
    MODEL = "/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro"
    DATE, COMMIT = "2026-10-29", "f6e1fa77d"
    RECORD = "622cbece7a489b55bf75879131b4456d21af34daa31e8294ba63625f6a354dfb"
#: On gpu-40 too, under nightly 20261026.
if GPU40:
    MODEL = "/shared/data/amd_int/models/DeepSeek-V4-Pro"
    DATE, COMMIT = "2026-10-29", "b71c071d3"
    RECORD = "300adea69f618f9da5fd7258c927c14000b2f247a1bad8c02bd74928b62ce787"
SUBJECT = {"model": MODEL, "width": 8, "source_root": f"commit:{COMMIT}"}
SUBJECT["record"] = f"sha256:{RECORD}"

#: aiter's commit in the stack each host's start and probe ran under (on 005
#: rocm/atom-dev sha256 4d44267b1d5a, nightly 20261026), and the day its cards
#: were read with no ptl directory.
AITER, PTL_DATE = "9205e1059", "2026-10-18"
if CRS005:
    AITER, PTL_DATE = "41d4d8d7e", "2026-10-29"
#: On gpu-40, rocm/atom-dev nightly 20261026 (sha256 4d44267b1d5a) with its
#: aiter 41d4d8d7e patched by `aiter_c90842add0.sh`, which commits upstream
#: aiter c90842add0's csrc on it as 09d256420 and rebuilds the modules that
#: compile it.
if GPU40:
    AITER, PTL_DATE = "09d256420", "2026-10-29"

#: Where the stack is read when the device start ran another: gpu-40's start
#: above ran the image's aiter, so its stack is the one `compass record` read
#: at a later start in a container `aiter_c90842add0.sh` built, run.json's
#: `stack`: (date, ATOM commit, model, TP width, run.json's sha256).
STACK = None
if GPU40:
    STACK = (
        "2026-11-09",
        "d698ff75e",
        "/shared/data/amd_int/models/DeepSeek-V4.1-Flash",
        4,
        "2930e1e2345ffbffe41d6a7158c1deab9ea3a3c8f5d123752588d699209e6670",
    )

DEVICE = {
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

#: On gpu-40, cards 0000:e5:00.0 and 0000:95:00.0 under nightly 20261026, its
#: aiter unpatched (`PROBED_AITER`). The spec carries them to `AITER`, saying
#: so in the peaks fragment's notes: every kernel the probe times (hipBLASLt,
#: `aiter.gemm_a4w4`, its own Triton GEMMs) is in a module the recipe leaves
#: byte-identical, since it rebuilds only the modules `aiter_c90842add0.sh`
#: lists.
PROBED_AITER = None
if GPU40:
    PROBED_AITER = "41d4d8d7e"
    PEAKS = (
        "2026-10-29",
        "b71c071d3",
        {
            "bf16": 1609038921361468.2,
            "fp8": 3361205421425982.5,
            "fp4": 5534379082687883.0,
            "bandwidth_curve": [
                [4096, 2608792614.3273535],
                [8192, 5215590387.848735],
                [16384, 10434366335.520044],
                [32768, 19588051820.511566],
                [65536, 38229081734.511765],
                [131072, 76369977669.6408],
                [262144, 152418432173.76987],
                [524288, 298199695070.9156],
                [1048576, 581902759044.4948],
                [2097152, 1084842857278.6284],
                [4194304, 1986616683069.1052],
                [8388608, 3068504544993.117],
                [16777216, 4095006064739.851],
                [33554432, 5151147111260.053],
                [67108864, 5843433335088.58],
                [134217728, 6064226650248.099],
                [268435456, 6136658132140.043],
                [536870912, 6376077711278.706],
                [1073741824, 6398494894990.093],
                [2147483648, 6462416692526.577],
                [4294967296, 6533795160761.386],
            ],
            "link": 61219294824.48995,
            "device.kernel_floor_s": [
                1.4669345021247865e-06,
                1.466374897956848e-06,
                1.4678549408912658e-06,
            ],
            "interconnect.intra_node.link_latency_s": [
                8.466782569885253e-07,
                8.374245047569274e-07,
                8.621324896812439e-07,
            ],
        },
        "020692760182d835943e2ac416f59df1abd2bb82cbe71870ce0c9f316cae436a",
    )

#: On 005, the network file of its rails into 013, which `rdma_link.py fit
#: BENCH RECORD` published; its notes say what each value was read from.
NETWORK = None
if CRS005:
    NETWORK = Path(__file__).resolve().parents[2] / "atom/compass/networks"
    NETWORK /= "crs005-013-ionic.json"

#: No probe fills these: mi308x_spec's assumed values less what the datasheet
#: states and what `PEAKS` and `NETWORK` fill. Its inter-node link is one
#: 400 Gb/s NIC's line rate here too.
DECLARED = copy.deepcopy(base.DECLARED)
for block, groups in DATASHEET.items():
    for group, values in groups.items():
        for key in values:
            DECLARED[block][group].pop(key, None)
if NETWORK:
    del DECLARED["interconnect"]["inter_node"]


def device_fragments() -> list[Fragment]:
    """The start's readings, the datasheet's peaks, the probed peaks and the
    assumed rest."""
    notes = f"one DeepSeek-V4-Pro {FLAGS} start at {COMMIT}"
    ptl = {"device": {"ptl": "disabled"}}
    datasheet = fragment(
        "datasheet", DATASHEET, "datasheet", "MI355X datasheet", date=DATE
    )
    assumed = fragment(
        "assumed", DECLARED, "assumed", "no probe fills these", date=DATE
    )
    probed, assumed = base.peaks(PEAKS, datasheet.values, assumed)
    if PROBED_AITER:
        carried = (
            f"; its record ran aiter {PROBED_AITER}, and every kernel it timed is "
            f"byte-identical under {AITER}: aiter_c90842add0.sh rebuilds no "
            "module that holds one"
        )
        notes_of = {"provenance.notes": probed.values["provenance.notes"] + carried}
        probed = Fragment(probed.source, {**probed.values, **notes_of})
    device, stack = DEVICE, []
    if STACK:
        date, root, model, width, record = STACK
        held = dict(DEVICE["device"])
        pinned = {"device": {"software_pinned_to": held.pop("software_pinned_to")}}
        device = {"device": held}
        subject = {"model": model, "width": width, "source_root": f"commit:{root}"}
        subject["record"] = f"sha256:{record}"
        how = (
            f"run.json's stack, one TP{width} start at {root} after aiter_c90842add0.sh"
        )
        stack = [fragment("stack", pinned, "probed", how, subject, date)]
    fragments = [
        fragment("device", device, "probed", notes, SUBJECT, DATE),
        *stack,
        datasheet,
        assumed,
        fragment("ptl", ptl, "probed", "no card has a ptl directory", date=PTL_DATE),
        probed,
    ]
    if NETWORK:
        network = json.loads(NETWORK.read_text()) | {"name": NAME}
        fragments.append(Fragment.from_mapping(network, NETWORK.stem))
    return fragments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    out = Path(__file__).resolve().parent / "machines" / f"{NAME}.json"
    parser.add_argument("--out", type=Path, default=out)
    args = parser.parse_args(argv)
    merged = merge(device_fragments())
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
