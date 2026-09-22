# SPDX-License-Identifier: MIT
"""What a probe that needs no device emits, and the reading that makes two
fragments comparable.

A probe measures part of a machine and emits a fragment: the fields it measured
plus the stanza saying which machine they are for, by whom, when and how. What
is here is the composing half of the tier that needs no device -- the tokenizer
rates arrive as an argument, from whoever measured them -- and a tokenizer
sweep needs no device, so it is the measurement most likely to be taken
somewhere other than the machine it is authored for -- a laptop, a build agent,
the login node in front of a cluster. That is not an abuse of it. Tokenization
is single-threaded, cache-resident integer work, so the rates transfer across
processors of comparable class, and refusing every spec not authored on its own
target would refuse most specs that will ever be written.

**So a fragment composed here writes down the processor this ran on, and that
is the whole of its second job.** `host.cpu.cores_physical` and
`host.cpu.cores_logical` are already required fields of the machine, so a
fragment that fills them in is claiming that the host that composed it is the
machine its stanza names. When it is not
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
A host that publishes no topology, or one that publishes a file holding nothing
a number can be read out of, gets a refusal naming what could not be read: a
count taken off an unreadable file is a reading with no content behind it, and
it would travel into a spec looking exactly like a measurement.

**The rates are the caller's, and so is the claim about how they were
obtained.** This function composes a fragment and reads a topology. It does not
encode or decode anything, so it cannot know whether the rates it was handed
were swept on a loaded tokenizer, copied out of an older spec or typed in -- and
`provenance.method` is what a merge refusal quotes and what a run artifact
carries as the numbers' source. A method stamped here would be this function
asserting a measurement it did not take, over data it received. So the caller
states it, with no default to fall through to, and the core counts read here
ride along under whatever the caller states about its own rates.

**What the entries are is checked here rather than several verbs later.** They
are read against the closed table a tokenizer entry has, the same one a spec is
read against, so an entry that is not a tokenizer, or one missing the derate its
two rates oblige, is refused while the probe that produced it is still the thing
in front of the author. Left to `validate`, the same entry is refused after a
merge, in a document whose other fields came from elsewhere, and the fragment
that carried it has to be worked back to.
"""

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from .fields import SCHEMA_VERSION
from .merge import Fragment
from .rules import Rule, SpecRefusal
from .tokenizers import table

#: Where the kernel publishes one directory per processor.
PROCESSORS = "/sys/devices/system/cpu"


def _unreadable(what: str) -> NoReturn:
    raise SpecRefusal(
        Rule.MEASURED,
        f"the processors of this host cannot be counted: {what}",
        "write `host.cpu.cores_physical` and `host.cpu.cores_logical` into the "
        "fragment by hand; the logical count is not the physical one, and on a "
        "host with simultaneous multithreading the two differ by the factor "
        "that makes the pair worth recording",
    )


def _reading(processor: pathlib.Path, name: str) -> int:
    """One topology number of one processor, or a refusal naming what it held."""
    try:
        published = (processor / "topology" / name).read_text()
    except OSError as unreadable:
        _unreadable(f"{processor.name} publishes no topology ({unreadable})")
    try:
        return int(published.strip())
    except ValueError:
        _unreadable(f"{processor.name} publishes {name} as {published!r}")


def cpu_counts(processors: str = PROCESSORS) -> dict[str, int]:
    """The `host.cpu` block of the machine this is running on, or a refusal."""
    root = pathlib.Path(processors)
    logical = sorted(path for path in root.glob("cpu[0-9]*") if path.name[3:].isdigit())
    if not logical:
        _unreadable(f"{root} lists no processor")
    cores = {
        (_reading(path, "physical_package_id"), _reading(path, "core_id"))
        for path in logical
    }
    return {"cores_physical": len(cores), "cores_logical": len(logical)}


def tokenizer_fragment(
    entries: Sequence[Mapping[str, Any]],
    *,
    machine: str,
    authored_by: str,
    date: str,
    method: str,
    source: str = "tokenizer",
    processors: str = PROCESSORS,
) -> Fragment:
    """A fragment of the entries a caller measured and the processor read here.

    `method` is how those entries were obtained, and it has no default because
    nothing in this function establishes one.
    """
    if not entries:
        raise SpecRefusal(
            Rule.TOKENIZER_IDENTITY,
            f"the tokenizer fragment for {machine!r} carries no tokenizer",
            "sweep a tokenizer and emit its entry; a fragment carrying only "
            "this host's core counts states nothing measured about the thing "
            "the probe is for",
        )
    # Read for its refusals: an entry that is not a tokenizer entry is refused
    # here, beside the caller that supplied it, rather than after a merge.
    table(entries)
    document = {
        "schema_version": SCHEMA_VERSION,
        "name": machine,
        "provenance": {"authored_by": authored_by, "date": date, "method": method},
        "host": {"cpu": cpu_counts(processors), "tokenizers": list(entries)},
    }
    return Fragment.from_mapping(document, source)
