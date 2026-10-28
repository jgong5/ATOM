# SPDX-License-Identifier: MIT
"""MI455's two machine specs: A0's from probes on one MI455 node, and B0's for
the performance machines, which no node here reaches.

    python3 scripts/compass/mi455_spec.py a0 [--model M ...] [--sweep-from DOC] [--out PATH]
    python3 scripts/compass/mi455_spec.py b0 [--out PATH]

MI455X's published numbers (`DATASHEET`) are both specs' peaks.

`mi455-a0`, the node's A0 sample cards: `probe_device.py`'s readings on two of
them (`PEAKS`) give each format's derate over the peaks, the bandwidth curve,
the kernel floor and the intra-node link's latency and derate, and its record
the device fragment (`DEVICE`): the arch, the capacity torch reads and the
stack, beside the host's cores. No card has a ptl directory. The HTTP intake
terms are mi355-gpu-45's, assumed, until a burst on MI455 fits them; what no
probe fills on any machine (the IPC costs, the inter-node link's latency and
derate, the router relay) is assumed as on mi355-gpu-45. `--model` and
`--sweep-from` work as `mi308x_spec.py`'s do; with neither the script sweeps
DeepSeek-V4.1-Flash's tokenizer.

`mi455-b0`, device `MI455X B0`, which no card reads: each attainable value is a
peak times the ratio mi355-gpu-45 probed, assumed, as `compass project` takes
them for a device never run. Every derate is MI355X's, each bandwidth-curve
point is MI355X's over its datasheet bandwidth times MI455X's, and where no
datasheet states a value MI355X's stands (`CARRIED`): its kernel floor and
intra-node link latency, its host and stack, and the inter-node link latency
and router relay it assumes. Its capacity is the datasheet's and its arch,
assumed, the A0 samples'; no rate, latency or stack A0's probe read reaches it.

Each writes `--out`, by default its file in `machines/`, when `validate`
passes, then prints `explain`'s rows grouped by origin.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mi308x_spec as base
from mi308x_spec import fragment, saved, swept

from atom.compass.spec import Fragment, MachineSpec, merge, validate
from atom.compass.validate.project import CARRIED

NAME = "mi455-a0"
base.NAME = NAME  # the shared fragment and sweep functions name this machine
B0 = "mi455-b0"
MACHINES = Path(__file__).resolve().parent / "machines"
#: The probed machine whose ratios and values B0 carries, and whose HTTP
#: intake terms A0 carries until it has its own.
MI355X = MACHINES / "mi355x-gpu45.json"
MODEL = "/apps/perzhang/models/DeepSeek-V4.1-Flash"
SHEET = "MI455X product page and brochure (07/26), read 2026-10-09"
#: B0's capacity, the datasheet's 432 GB of HBM4 as GiB (`CAPACITY`).
B0_CAPACITY = 432 * 2**30
CAPACITY = (
    "its 432 GB of HBM4 as GiB, since MI355X's 288 GB reads 288 GiB less 16 MiB "
    "at a real start and MI308X's 192 GB reads 192 GiB less 16 MiB"
)

#: AMD's MI455X product page and brochure (07/26), read 2026-10-09: dense
#: matrix-core peaks, HBM4 bandwidth, the four GPUs of a compute tray and the
#: 72 its switched UALoE fabric joins, one GPU's scale-up rate and its NICs'
#: scale-out rate, each per direction.
DATASHEET = {
    "device": {
        "name": "MI455X",
        "count_per_node": 4,
        "memory": {"bandwidth_bytes_per_s": 23.3e12},
        "compute": {
            "bf16_flops": 5.033e15,
            "fp8_flops": 20.133e15,
            "fp4_flops": 40.265e15,
        },
    },
    "interconnect": {
        "intra_node": {
            "topology": "switched",
            "domain_devices": 72,
            "link_bandwidth_bytes_per_s": 1.8e12,
        },
        "inter_node": {"link_bandwidth_bytes_per_s": 3.0e11},
    },
}

#: `probe_device.py`'s readings on cards 0001:04:00.0 and 0002:04:00.0 of the
#: node, gfx1250 A0 samples of 192 compute units, neither with a ptl
#: directory, under rocm/fw-bringup-mlperf:gfx1250-atom-20261026-dsv41
#: (sha256:fd7450fde2da), the stack `DEVICE` pins, at the commit named; the
#: record is its OUT. Each format's rate is its Triton GEMM's, which ran past
#: hipBLASLt's in all three; aiter's MXFP4 GEMM did not compile. Each derate is
#: a reading over `DATASHEET`'s peak, so it also holds the samples' 192 of the
#: 256 compute units AMD counts for MI455X.
PEAKS = (
    "2026-10-27",
    "dc5e8efa6",
    {
        "bf16": 1328645931957327.5,
        "fp8": 1447678251827510.2,
        "fp4": 2780799929426501.0,
        "bandwidth_curve": [
            [4096, 1530666480.1579611],
            [8192, 3060989328.727323],
            [16384, 6121244111.007896],
            [32768, 12244140021.858593],
            [65536, 24485718013.116013],
            [131072, 48974729786.66798],
            [262144, 104017760859.1218],
            [524288, 208027305262.93582],
            [1048576, 474430357731.4853],
            [2097152, 909953098725.0911],
            [4194304, 1564486569957.3643],
            [8388608, 2747429481067.666],
            [16777216, 5106961425379.4],
            [33554432, 7379491450234.512],
            [67108864, 9802761677897.846],
            [134217728, 12245916103059.629],
            [268435456, 13789455773461.098],
            [536870912, 14891726350520.275],
            [1073741824, 15201029898451.355],
            [2147483648, 15644687861213.988],
            [4294967296, 15777066512305.184],
        ],
        "link": 629989898092.7966,
        "device.kernel_floor_s": [
            1.3443179726600648e-06,
            1.3400887489318846e-06,
            1.349955356121063e-06,
        ],
        "interconnect.intra_node.link_latency_s": [
            2.6227587461471556e-06,
            2.5976027250289913e-06,
            2.6680062055587767e-06,
        ],
    },
    "d7dfcff22ef49192be8ba8f3a2a4b4fd5e475835fb9edf1ab412863ed1a4aceb",
)

#: What `PEAKS`'s record read on card 0 (the arch and the total torch reads, as
#: an engine start reads it) and the stack it ran under; the host is the node's.
DEVICE = {
    "host": {"cpu": {"cores_physical": 128, "cores_logical": 256}},
    "device": {
        "arch": "gfx1250",
        "memory": {"capacity_bytes": 309237645312},
        "software_pinned_to": {
            "rocm": "10.1.0",
            "aiter": "99e4450b6",
            "rccl": "2.30.4",
            "torch": "2.11.0+rocm10.1.0a20260811-bkc.20260820",
        },
    },
}

#: No probe fills these: mi308x_spec's assumed values less what the datasheet
#: states and what `PEAKS` fills.
DECLARED = copy.deepcopy(base.DECLARED)
for block, groups in DATASHEET.items():
    for group, values in groups.items():
        for key in values if isinstance(values, dict) else ():
            DECLARED[block][group].pop(key, None)


def a0_fragments() -> list[Fragment]:
    """A0's: the probe record's device, the datasheet's peaks, the probed
    peaks, the cards' PTL state, MI355X's intake terms and the assumed rest."""
    date, root, _, record = PEAKS
    subject = {"source_root": f"commit:{root}", "record": f"sha256:{record}"}
    notes = f"probe_device.py's record at {root}"
    datasheet = fragment("datasheet", DATASHEET, "datasheet", SHEET, date=date)
    assumed = fragment(
        "assumed", DECLARED, "assumed", "no probe fills these", date=date
    )
    probed, assumed = base.peaks(PEAKS, datasheet.values, assumed)
    mi355x = MachineSpec.from_mapping(json.loads(MI355X.read_text()))
    intake = {
        "host": {
            "http_intake": {
                k: mi355x.value(f"host.http_intake.{k}")
                for k in ("fixed_s", "per_byte_s")
            }
        }
    }
    carried = f"{MI355X.stem}'s, until a burst on MI455 fits them"
    ptl = {"device": {"ptl": "disabled"}}
    return [
        fragment("device", DEVICE, "probed", notes, subject, date),
        datasheet,
        assumed,
        fragment("ptl", ptl, "probed", "no card has a ptl directory", date=date),
        fragment("intake", intake, "assumed", carried, date=date),
        probed,
    ]


def b0_fragment(
    source: str, body: dict, origin: str, notes: str, date: str
) -> Fragment:
    """One fragment of B0's spec, its notes saying where its values came from."""
    stanza = {
        "authored_by": base.AUTHOR,
        "date": date,
        "origin": origin,
        "notes": notes,
    }
    document = {"schema_version": 1, "name": B0, "provenance": stanza, **body}
    return Fragment.from_mapping(document, source)


def b0_fragments() -> list[Fragment]:
    """B0's: the datasheet's peaks, B0's name and arch, MI355X's ratios on the
    peaks and its values where no datasheet states one."""
    mi355x = MachineSpec.from_mapping(json.loads(MI355X.read_text()))
    peak = DATASHEET["device"]["memory"]["bandwidth_bytes_per_s"]
    scale = peak / mi355x.value("device.memory.bandwidth_bytes_per_s")
    ratios = {p: v for p, v in mi355x.values.items() if "derate" in p.split(".")}
    ratios["device.memory.bandwidth_curve"] = [
        [size, rate * scale]
        for size, rate in mi355x.value("device.memory.bandwidth_curve")
    ]
    carried = {p: v for p, v in mi355x.values.items() if p.startswith(CARRIED)}
    sheet = copy.deepcopy(DATASHEET)
    del sheet["device"]["name"]
    sheet["device"]["memory"]["capacity_bytes"] = B0_CAPACITY
    b0 = {"device": {"name": "MI455X B0", "arch": "gfx1250"}}
    never_ran = f"no B0 card ran: {MI355X.stem}'s"
    date = "2026-10-28"
    return [
        b0_fragment(
            "datasheet", sheet, "datasheet", f"{SHEET}; {CAPACITY}", "2026-10-09"
        ),
        b0_fragment(
            "b0",
            b0,
            "assumed",
            "the B0 performance machines, which no card here reads: MI455X "
            "named for their stepping, at the arch the A0 samples read",
            date,
        ),
        b0_fragment(
            "ratios",
            ratios,
            "assumed",
            f"{never_ran} derates, and its bandwidth curve over its datasheet "
            "bandwidth times MI455X's",
            date,
        ),
        b0_fragment(
            "carried",
            carried,
            "assumed",
            f"{never_ran} values where no datasheet states one: its probed kernel "
            "floor and intra-node link latency, its host and stack, and the "
            "inter-node link latency and router relay it assumes",
            date,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stepping", choices=("a0", "b0"))
    parser.add_argument("--out", type=Path, help="default machines/mi455-STEPPING.json")
    parser.add_argument(
        "--model", action="append", help="a0; repeats; default " + MODEL
    )
    parser.add_argument("--sweep-from", type=Path, help="a0: a document to reuse")
    args = parser.parse_args(argv)
    if args.stepping == "b0":
        if args.model or args.sweep_from:
            parser.error("b0 sweeps no tokenizer: it carries mi355-gpu-45's host")
        fragments = b0_fragments()
    else:
        sweeps = [saved(args.sweep_from)] if args.sweep_from else []
        if args.model or not sweeps:
            sweeps += swept(args.model or [MODEL])
        fragments = [*sweeps, *a0_fragments()]
    merged = merge(fragments)
    verdict = validate(merged.document)
    print(verdict)
    if not verdict.ok:
        return 1
    base.refuse_unbanded(MachineSpec.from_mapping(merged.document))
    out = args.out or MACHINES / f"mi455-{args.stepping}.json"
    out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(base.report(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
