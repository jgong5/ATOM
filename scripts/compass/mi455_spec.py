# SPDX-License-Identifier: MIT
"""The MI455X machine spec, for a device no engine has run on.

    python3 scripts/compass/mi455_spec.py [--out PATH]

MI455X's published numbers (`DATASHEET`) are its peaks. Each attainable value
is that peak times the ratio mi355-gpu-45 probed (`MI355X`), assumed: every
derate is MI355X's, and each bandwidth-curve point is MI355X's over its
datasheet bandwidth, times MI455X's. What no datasheet states, MI355X's value
carried over is assumed (`CARRIED`): its probed kernel floor and intra-node
link latency, its host, its stack, and the inter-node link latency and router
relay it assumes itself. Writes the merged document to `--out` when `validate`
passes, then prints `explain`'s rows grouped by origin.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mi308x_spec as base

from atom.compass.spec import Fragment, MachineSpec, merge, validate

NAME = "mi455"
MACHINES = Path(__file__).resolve().parent / "machines"
#: The probed device whose ratios and values MI455 carries.
MI355X = MACHINES / "mi355x-gpu45.json"
DATE = "2026-10-21"

#: AMD's MI455X product page and brochure (07/26), read 2026-10-09: dense
#: matrix-core peaks, HBM4 bandwidth, the four GPUs of a compute tray and the
#: 72 its switched UALoE fabric joins, one GPU's scale-up rate and its NICs'
#: scale-out rate, each per direction. Its 432 GB of HBM4 is binary: MI355X's
#: 288 GB reads 309220868096 B at a real start, 288 GiB less 16 MiB, and
#: MI308X's 192 GB reads 192 GiB less 16 MiB.
DATASHEET = {
    "device": {
        "name": "MI455X",
        "count_per_node": 4,
        "memory": {"capacity_bytes": 432 * 2**30, "bandwidth_bytes_per_s": 23.3e12},
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

#: What MI455 takes from MI355X as it stands, no datasheet stating it.
CARRIED = (
    "host.",
    "device.kernel_floor_s",
    "device.software_pinned_to.",
    "interconnect.intra_node.link_latency_s",
    "interconnect.inter_node.link_latency_s",
    "interconnect.router_relay_s",
)


def fragment(source: str, body: dict, origin: str, notes: str, date=DATE) -> Fragment:
    stanza = {
        "authored_by": base.AUTHOR,
        "date": date,
        "origin": origin,
        "notes": notes,
    }
    document = {"schema_version": 1, "name": NAME, "provenance": stanza, **body}
    return Fragment.from_mapping(document, source)


def device_fragments() -> list[Fragment]:
    """The datasheet's peaks, MI355X's ratios on them, and MI355X's values
    carried over, each fragment's notes saying where its values came from."""
    mi355x = MachineSpec.from_mapping(json.loads(MI355X.read_text()))
    peak = DATASHEET["device"]["memory"]["bandwidth_bytes_per_s"]
    scale = peak / mi355x.value("device.memory.bandwidth_bytes_per_s")
    ratios = {p: v for p, v in mi355x.values.items() if "derate" in p.split(".")}
    ratios["device.memory.bandwidth_curve"] = [
        [size, rate * scale]
        for size, rate in mi355x.value("device.memory.bandwidth_curve")
    ]
    carried = {p: v for p, v in mi355x.values.items() if p.startswith(CARRIED)}
    sheet = (
        "MI455X product page and brochure (07/26), read 2026-10-09; its 432 GB "
        "of HBM4 as GiB, since MI355X's 288 GB reads 288 GiB less 16 MiB at a "
        "real start and MI308X's 192 GB reads 192 GiB less 16 MiB"
    )
    arch = "LLVM's AMDGPUUsage lists gfx1250 for MI455X; AMD has not confirmed it"
    never_ran = f"MI455X never ran: {MI355X.stem}'s"
    return [
        fragment("datasheet", DATASHEET, "datasheet", sheet, date="2026-10-09"),
        fragment("arch", {"device.arch": "gfx1250"}, "assumed", arch),
        fragment(
            "ratios",
            ratios,
            "assumed",
            f"{never_ran} derates, and its bandwidth curve over its datasheet "
            "bandwidth times MI455X's",
        ),
        fragment(
            "carried",
            carried,
            "assumed",
            f"{never_ran} values where no datasheet states one: its probed kernel "
            "floor and intra-node link latency, its host and stack, and the "
            "inter-node link latency and router relay it assumes",
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=MACHINES / f"{NAME}.json")
    args = parser.parse_args(argv)
    merged = merge(device_fragments())
    verdict = validate(merged.document)
    print(verdict)
    if not verdict.ok:
        return 1
    args.out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(base.report(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
