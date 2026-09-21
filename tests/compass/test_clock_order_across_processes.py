# SPDX-License-Identifier: MIT
"""The order the registry serves participants in is the same in every process.

This is the one claim in `atom.compass.clock` that cannot be tested in the
process running the test. Python randomises string hashing per process, so two
registries built inside one interpreter share one seed: a registry that iterated
a `set` of names would produce the same order twice and a same-process test
would pass against precisely the defect it exists to catch. The interpreters
below are given different `PYTHONHASHSEED` values, so the seed is the variable.

Each child also builds a plain `set` of the same names, in the same order in
every child, and prints the order it iterates in. That is the control, and it is
the reason the result reads as a demonstration rather than an assertion: the
same names, in the same children, come out in a different order per seed through
a `set` and in one order through the registry. The seed is the only thing that
differs between those `set` lines -- they are built from the unrotated list on
purpose, because insertion order moves set order a little by itself and a
control varying two things at once attributes nothing. The `set` here is
deliberate and belongs to the control; the package under test builds none, which
its own source check covers.

Arrival order is varied separately -- each child registers the names rotated by
a different amount -- so an implementation that simply preserved insertion order
fails this too.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Names chosen to be the ones a real deployment uses, and long enough that a
# 12-element table is nowhere near collision-free under a changed seed.
NAMES = [
    "traffic-source",
    "prefill-0",
    "prefill-1",
    "decode-0",
    "decode-1",
    "pp-stage-0",
    "pp-stage-1",
    "pp-stage-2",
    "pp-stage-3",
    "replica-a",
    "replica-b",
    "replica-c",
]

# Three seeds, so "the two that differ" is not the only evidence available, and
# fixed rather than random so a failure is reproducible from the test name.
SEEDS = ("1", "2", "3")

CHILD = """
import os, sys
import atom
from atom.compass.clock import LpId, LpRegistry

names = sys.argv[2].split(",")
rotate = int(sys.argv[1])
arrival = names[rotate:] + names[:rotate]

registry = LpRegistry()
for name in arrival:
    registry.register(LpId(name))

print("seed", os.environ.get("PYTHONHASHSEED", "<unset>"))
print("hash", hash(names[0]))
print("module", atom.__file__)
print("arrival", " ".join(arrival))
print("registry", " ".join(str(lp_id) for lp_id in registry.ids()))
# Built from `names`, not from `arrival`: the seed is then the only thing that
# differs between children, so a difference in this line is attributable to it
# and to nothing else. Insertion order does move set order a little on its own,
# which is exactly the confound worth not having in the control.
print("set", " ".join(set(names)))
"""


def _child(seed, rotate):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = str(REPO)
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(rotate), ",".join(NAMES)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, f"seed {seed} child failed:\n{done.stderr}"
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    return dict(line.split(" ", 1) for line in lines)


@pytest.fixture(scope="module")
def runs():
    # One child per seed, each registering in a different rotation. Spawned once
    # for the module: the import is the expensive part and it is the same import
    # for every assertion below.
    return [_child(seed, rotate) for rotate, seed in enumerate(SEEDS)]


def test_the_children_really_did_get_different_hash_seeds(runs):
    # Without this the rest of the file could pass on three identically-seeded
    # interpreters, which is the vacuous version of the test.
    seeds = [run["seed"] for run in runs]
    hashes = [run["hash"] for run in runs]
    assert seeds == list(SEEDS)
    assert len(dict.fromkeys(hashes)) == len(SEEDS), (
        f"the same name hashed identically under {seeds}: {hashes}. Either "
        "hash randomisation is disabled in this interpreter or the seed did "
        "not reach the child, and nothing below is evidence of anything."
    )


def test_every_child_ran_against_the_tree_under_test(runs):
    # A demonstration whose value is that its output can be read should say
    # which tree produced it. PYTHONPATH wins over site-packages, but an
    # installed `atom` on the path is the kind of thing that turns a green run
    # into evidence about someone else's code.
    for run in runs:
        assert run["module"].startswith(str(REPO)), (
            f"seed {run['seed']} resolved atom at {run['module']}, "
            f"which is not under {REPO}"
        )


def test_the_children_really_did_register_in_different_orders(runs):
    arrivals = [run["arrival"] for run in runs]
    assert len(dict.fromkeys(arrivals)) == len(SEEDS), arrivals


def test_the_total_order_is_the_same_in_every_process(runs):
    expected = " ".join(sorted(NAMES))
    orders = [run["registry"] for run in runs]
    assert (
        len(dict.fromkeys(orders)) == 1
    ), "the registry gave a different order per process: " + "; ".join(
        f"seed {run['seed']}: {run['registry']}" for run in runs
    )
    assert orders[0] == expected


def test_a_set_of_the_same_names_does_not_survive_a_change_of_seed(runs):
    # The control, and the reason the result above is a measurement. These are
    # the same twelve names in the same three children; only the container they
    # went through differs.
    set_orders = [run["set"] for run in runs]
    assert len(dict.fromkeys(set_orders)) > 1, (
        "every seed iterated the set identically, so this run demonstrates "
        f"nothing about the registry: {set_orders}"
    )
