# SPDX-License-Identifier: MIT
"""What a probe that needs no device emits, and the reading that makes two
fragments comparable.

A probe measures part of a machine and emits a fragment: the fields it measured
plus the stanza saying which machine they are for, by whom, when and how. The
tokenizer sweep needs no device, so it is the probe most likely to be run
somewhere other than the machine it is authored for -- a laptop, a build agent,
the login node in front of a cluster. That is not an abuse of it. Tokenization
is single-threaded, cache-resident integer work, so the rates transfer across
processors of comparable class, and refusing every spec not authored on its own
target would refuse most specs that will ever be written.

**So this probe writes down the processor it ran on, and that is the whole of
its second job.** `host.cpu.cores_physical` and `host.cpu.cores_logical` are
already required fields of the machine, so a probe that fills them in is
claiming that the host it ran on is the machine its stanza names. When it is not
-- the tokenizer measured on the laptop, the device fields measured on the node,
both authored for the node -- the two fragments then state different numbers for
one field of one machine, and the merge refuses the pair with both readings and
both stanzas. Before this they shared no field at all: nothing in a fragment's
shape says where it was measured, so nothing contradicted anything, and the pair
combined into a document that describes neither host. Two fields that were
already required, and nothing added to the schema, are what turn that silence
into a refusal.

This does not make the transfer illegitimate, and it does not record where the
probe ran -- no field of a fragment does that. A tokenizer measured on a
comparable processor is still the intended use. What changes is that an author
who wants it now has to write the target's own core counts into the document,
rather than have the question go unasked.

**The counts come from the kernel's per-processor topology, and are refused
rather than guessed.** The physical count is the number of distinct
package-and-core pairs; the logical count is the number of processors. They
differ by the simultaneous-multithreading factor, so deriving either from the
other would be wrong by exactly the factor that makes the pair worth recording.
A host that publishes no topology gets a refusal naming what could not be read,
because a count invented here would travel into a spec as a measurement.
"""

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from .fields import SCHEMA_VERSION
from .merge import Fragment
from .rules import Rule, SpecRefusal

#: Where the kernel publishes one directory per processor.
PROCESSORS = "/sys/devices/system/cpu"
#: How a fragment from this probe says its numbers were obtained.
METHOD = "probed"


def _unreadable(what: str) -> NoReturn:
    raise SpecRefusal(
        Rule.SHAPE,
        f"the processors of this host cannot be counted: {what}",
        "write `host.cpu.cores_physical` and `host.cpu.cores_logical` into the "
        "fragment by hand; the logical count is not the physical one, and on a "
        "host with simultaneous multithreading the two differ by the factor "
        "that makes the pair worth recording",
    )


def cpu_counts(processors: str = PROCESSORS) -> dict[str, int]:
    """The `host.cpu` block of the machine this is running on, or a refusal."""
    root = pathlib.Path(processors)
    logical = sorted(path for path in root.glob("cpu[0-9]*") if path.name[3:].isdigit())
    if not logical:
        _unreadable(f"{root} lists no processor")
    cores = set()
    for path in logical:
        topology = path / "topology"
        try:
            package = (topology / "physical_package_id").read_text()
            core = (topology / "core_id").read_text()
        except OSError as unreadable:
            _unreadable(f"{path.name} publishes no topology ({unreadable})")
        cores.add((package.strip(), core.strip()))
    return {"cores_physical": len(cores), "cores_logical": len(logical)}


def tokenizer_fragment(
    entries: Sequence[Mapping[str, Any]],
    *,
    machine: str,
    authored_by: str,
    date: str,
    source: str = "tokenizer",
    processors: str = PROCESSORS,
) -> Fragment:
    """The tokenizer sweep's fragment: the rates, and the processor they ran on."""
    if not entries:
        raise SpecRefusal(
            Rule.TOKENIZER_IDENTITY,
            f"the tokenizer probe for {machine!r} measured no tokenizer",
            "sweep a tokenizer and emit its entry; a fragment carrying only "
            "this host's core counts states nothing measured about the thing "
            "the probe is for",
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "name": machine,
        "provenance": {"authored_by": authored_by, "date": date, "method": METHOD},
        "host": {"cpu": cpu_counts(processors), "tokenizers": list(entries)},
    }
    return Fragment.from_mapping(document, source)
