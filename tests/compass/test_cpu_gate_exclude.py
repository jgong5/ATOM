# SPDX-License-Identifier: MIT
"""`scripts/compass/cpu_gate_exclude.txt`: the list the CPU test gate trusts.

The list names the ATOM tests that reach the driver, so `gate_cpu.sh` can skip
them and stay green without a GPU.

It has two sections, and the split is the point. The GENERATED section is
produced by iterating collection to a fixed point and is reproducible from the
tree. The MANUAL section is for files that *collect* cleanly and fail later on a
driver call -- invisible to a generator that greps collection errors, so they
cannot be derived and must be asserted, with the failure text that justifies
each one. Before the split, four entries had been hand-added to a file whose
header read "never hand-edit", and the list could not be reproduced from the
tree it described. These tests exist so that cannot recur silently.

The failure being guarded against is drift, and drift here is silent in the
direction that matters. A renamed or deleted ATOM test leaves a stale
`--ignore=` that pytest accepts without complaint, so the entry stops excluding
anything and the gate starts erroring at collection on a file nobody listed --
which reads as "your change broke collection".

What is *not* asserted: that every listed file genuinely needs a driver. Proving
that means collecting each one in a driver-free container, which is
`regen_cpu_gate_exclude.sh`'s job, not a unit test's.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXCLUDE = REPO / "scripts" / "compass" / "cpu_gate_exclude.txt"
TRIGGERS = REPO / "scripts" / "compass" / "gpu_gate_triggers.txt"

GEN_BEGIN, GEN_END = "# BEGIN GENERATED", "# END GENERATED"
MAN_BEGIN, MAN_END = "# BEGIN MANUAL", "# END MANUAL"


def _lines(path):
    return path.read_text().splitlines()


def _paths(lines):
    return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _section(begin, end):
    """Lines strictly between two markers, markers excluded.

    Returns [] when a marker is missing rather than raising, so the dedicated
    marker test reports that failure once instead of every test erroring.
    """
    lines = _lines(EXCLUDE)
    if begin not in lines or end not in lines:
        return []
    return lines[lines.index(begin) + 1 : lines.index(end)]


def _entries():
    return _paths(_lines(EXCLUDE))


def test_the_list_exists_and_is_not_empty():
    assert EXCLUDE.is_file(), f"{EXCLUDE} is missing; run regen_cpu_gate_exclude.sh"
    assert _entries(), "an empty exclusion list means the CPU gate runs driver tests"


def test_both_section_markers_are_present():
    # gate_cpu.sh tolerates their absence -- it reads every non-comment line --
    # but regen_cpu_gate_exclude.sh refuses without them, so losing a marker
    # turns a reproducible list back into a hand-maintained one at the next
    # regeneration, quietly.
    lines = _lines(EXCLUDE)
    for marker in (GEN_BEGIN, GEN_END, MAN_BEGIN, MAN_END):
        assert marker in lines, f"{marker} missing from {EXCLUDE.name}"
    assert lines.index(GEN_BEGIN) < lines.index(GEN_END) < lines.index(MAN_BEGIN)
    assert lines.index(MAN_BEGIN) < lines.index(MAN_END)


@pytest.mark.parametrize("entry", _entries())
def test_every_excluded_path_still_exists(entry):
    # Parametrized so a rename names the offending file, rather than reporting
    # "1 of 29 paths is wrong" and leaving the reader to diff.
    assert (REPO / entry).is_file(), (
        f"{entry} is listed but does not exist -- ATOM renamed or removed it. "
        "Regenerate with scripts/compass/regen_cpu_gate_exclude.sh."
    )


def test_no_plugin_tests_are_listed():
    # gate_cpu.sh excludes tests/plugin/ by directory, for a different reason:
    # sglang and vllm are in neither image. Listing one here too would assert it
    # is a driver problem, and would survive the day the images gain sglang.
    strays = [e for e in _entries() if e.startswith("tests/plugin/")]
    assert not strays, f"excluded by directory already, drop from the list: {strays}"


def test_the_generated_section_is_sorted_and_unique():
    # The generator writes `sort -u`. Anything else means it was edited by hand,
    # which is how the list gets one batch of collection errors and stops.
    # Asserted per section, not over the file: the manual entries are appended
    # after the generated block and are not expected to interleave with it.
    entries = _paths(_section(GEN_BEGIN, GEN_END))
    assert entries, "the generated section is empty; run regen_cpu_gate_exclude.sh"
    assert entries == sorted(set(entries)), "not sorted+unique: regenerate, do not edit"


def test_the_manual_section_is_sorted_and_unique():
    entries = _paths(_section(MAN_BEGIN, MAN_END))
    assert entries == sorted(set(entries)), f"sort the manual entries: {entries}"


def test_no_path_appears_in_both_sections():
    # A duplicate is harmless to pytest and fatal to the reader: it makes the
    # generated section look like it discovered something the generator cannot
    # see, which is exactly the confusion the split removes.
    gen = set(_paths(_section(GEN_BEGIN, GEN_END)))
    man = set(_paths(_section(MAN_BEGIN, MAN_END)))
    assert not gen & man, f"listed twice: {sorted(gen & man)}"


@pytest.mark.parametrize("entry", _paths(_section(MAN_BEGIN, MAN_END)))
def test_every_manual_entry_states_why(entry):
    # A manual entry is an assertion no script can check, so the reason is the
    # only evidence it carries. Without one it is indistinguishable from a test
    # somebody found inconvenient -- which is what three of the four original
    # hand-added entries turned out to be: they passed on CPU.
    section = _section(MAN_BEGIN, MAN_END)
    above = section[: section.index(entry)]
    reason = [ln for ln in reversed(above) if ln.strip()]
    assert reason and reason[0].lstrip().startswith("#"), (
        f"{entry} has no comment above it. State the observed failure -- the "
        "error text and where it was measured -- or move it to the generated "
        "section by making regen_cpu_gate_exclude.sh find it."
    )


@pytest.mark.parametrize("entry", _paths(_lines(TRIGGERS)) if TRIGGERS.is_file() else [])
def test_every_trigger_path_still_exists(entry):
    # Same drift, one file over. A trigger naming a moved module matches
    # nothing, and gate_cpu.sh then reports "gpu: not required" for a diff that
    # needs it -- a false green, which is worse than the stale --ignore= above.
    target = REPO / entry
    assert target.is_dir() if entry.endswith("/") else target.is_file(), (
        f"{entry} is listed in {TRIGGERS.name} but does not exist. "
        "Regenerate with scripts/compass/regen_gpu_gate_triggers.sh."
    )
