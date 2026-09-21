# SPDX-License-Identifier: MIT
"""The set-iteration check, shown firing on a module written to break it.

A check that has never been seen failing is not a check, so the pass here is run
against a module that reads sets in every shape it claims to find, and the whole
of its report is pinned -- the text a future reader meets is the thing being
delivered, not the exit code.

The same module is the record of what the pass cannot see. Seven of its methods
iterate a set through a shape that is not reported: four because the set is not
recognised -- handed back by a call the module does not annotate, arriving in an
unannotated parameter, pulled out of a list, or a subclass of `set` -- and three
because the read is not, which is a separate class of gap with its own entries.
They are asserted **absent** from the report, in the same place as the ones that
are present, because a limitation that lives only in a docstring is a limitation
nobody re-checks. Absent is what was measured; complete is not claimed.

Two false positives are pinned here as well, and they matter more than a miss.
A name rebound to something that is not a set stops being one, so a reader who
applies the remedy stops being told to apply it; and a name is a set only inside
the scope that bound it, so a short name in one function does not fail the gate
in another.

The clock-source pass is CA-5's and is not rebuilt here, but it has two
comparisons of the same shape and both are settled here. Its allow-list decides
which files are excused, and the question is how many modules one entry stands
for. Its read list decides which calls are reads at all, and that comparison was
unanchored in the way the allow-list no longer is: a name merely ending in a
listed one collected the whole list. Both now compare on a boundary, and what
the read list still cannot see -- a clock bound where the parser does not follow
and called later -- is asserted absent rather than described.
"""

import os

import pytest

import atom.compass
from atom.compass.detect.clock_source import DEFAULT_ALLOW_LIST, ClockSourceLint
from atom.compass.detect.set_iteration import SetIterationLint

#: The tree the rule applies to.
SIMULATED_PATH = os.path.dirname(atom.compass.__file__)

#: A step loop that reads its participants out of a set every way the pass
#: claims to see, and seven more it does not. Everything here runs; nothing
#: here raises.
SIMULATED_PATH_READING_SETS = '''# SPDX-License-Identifier: MIT
"""A step loop that decides what to charge by reading a set."""

import functools
import itertools
from dataclasses import dataclass, field


def ready_for(names):
    return set(names)


def ready_now() -> set[str]:
    return set()


class SubclassedSet(set):
    pass


class Step:
    slots: set[str] = set()

    def __init__(self, participants):
        self.ready = set(participants)
        self.done: set[str] = set()
        self.held = [set(participants)]

    def charge(self, now):
        table = []
        for lp_id in self.ready:
            table.append((lp_id, now))
        return table

    def names(self):
        return [lp_id.upper() for lp_id in self.done]

    def first(self):
        head, *rest = self.ready
        return head

    def joined(self):
        return ", ".join(self.done)

    def fresh(self, other):
        for lp_id in self.ready - other:
            yield lp_id

    def merged(self, other):
        for lp_id in self.ready.union(other):
            yield lp_id

    def copied(self):
        held = self.done.copy()
        for lp_id in held:
            yield lp_id

    def viewed(self, table):
        for lp_id in table.keys() & self.ready:
            yield lp_id

    def listed(self):
        return list(self.done)

    def spread(self):
        return [*self.ready]

    def mapped(self):
        return list(map(str.upper, self.done))

    def filtered(self):
        return list(filter(None, self.done))

    def keyed(self):
        return dict.fromkeys(self.ready)

    def drained(self):
        while self.ready:
            yield self.ready.pop()

    def printed(self):
        return f"waiting on {self.ready}"

    def chosen(self, cost):
        return max(self.done, key=cost)

    def slotted(self):
        return [lp_id for lp_id in self.slots]

    def annotated(self):
        return [lp_id for lp_id in ready_now()]

    def sorted_out(self):
        return sorted(self.ready)

    def holds(self, lp_id):
        return lp_id in self.ready and len(self.ready) > 0

    def counted(self, table):
        return [lp_id for lp_id in table.keys()]

    def returned(self, names):
        return [lp_id for lp_id in ready_for(names)]

    def passed(self, arriving):
        return [lp_id for lp_id in arriving]

    def nested(self):
        return [lp_id for lp_id in self.held[0]]

    def subclassed(self, names):
        return [lp_id for lp_id in SubclassedSet(names)]

    def chained(self, other):
        return list(itertools.chain(self.ready, other))

    def reduced(self, combine):
        return functools.reduce(combine, self.done)

    def tie_broken(self, cost):
        return sorted(self.ready, key=cost)


@dataclass
class Batch:
    members: set[str] = field(default_factory=set)

    def priced(self):
        return [lp_id for lp_id in self.members]
'''

#: What the pass says about that module, whole. The line numbers are read off
#: the text above so the two cannot drift apart silently.
_LINES = SIMULATED_PATH_READING_SETS.splitlines()


def line_of(fragment):
    """The line the fragment is on, one-based, refusing an ambiguous one."""
    hits = [index + 1 for index, line in enumerate(_LINES) if fragment in line]
    assert len(hits) == 1, f"{fragment!r} is on {len(hits)} lines, not one"
    return hits[0]


EXPECTED_READS = (
    ("for lp_id in self.ready:", "for-loop over self.ready", "charge"),
    ("lp_id.upper() for lp_id in self.done", "comprehension over self.done", "names"),
    ("head, *rest = self.ready", "unpacking of self.ready", "first"),
    ('", ".join(self.done)', "join() over self.done", "joined"),
    ("self.ready - other", "for-loop over self.ready - other", "fresh"),
    (
        "self.ready.union(other)",
        "for-loop over self.ready.union(other)",
        "merged",
    ),
    ("for lp_id in held:", "for-loop over held", "copied"),
    (
        "table.keys() & self.ready",
        "for-loop over table.keys() & self.ready",
        "viewed",
    ),
    ("return list(self.done)", "list() over self.done", "listed"),
    ("[*self.ready]", "unpacking of self.ready", "spread"),
    ("map(str.upper, self.done)", "map() over self.done", "mapped"),
    ("filter(None, self.done)", "filter() over self.done", "filtered"),
    ("dict.fromkeys(self.ready)", "fromkeys() over self.ready", "keyed"),
    ("yield self.ready.pop()", "pop() from self.ready", "drained"),
    ('f"waiting on {self.ready}"', "formatted string of self.ready", "printed"),
    ("max(self.done, key=cost)", "max(key=...) over self.done", "chosen"),
    ("lp_id for lp_id in self.slots", "comprehension over self.slots", "slotted"),
    ("lp_id for lp_id in ready_now()", "comprehension over ready_now()", "annotated"),
    ("lp_id for lp_id in self.members", "comprehension over self.members", "priced"),
)

#: The shapes that are not reported, by the method they are in. Each is a real
#: iteration of a real set. The first four are a set the pass does not
#: recognise; the last three are a set it recognises read in a way it does not.
BEYOND_THE_PARSER = (
    "returned",
    "passed",
    "nested",
    "subclassed",
    "chained",
    "reduced",
    "tie_broken",
)


#: Every shape the pass claims to report, one module each. This is the
#: regression set: the claim is that all of these fire, and a change to the
#: recognition rules that quietly drops one fails here rather than in a review.
FORMS_THAT_FIRE = {
    "for": "def f(s: set[str]):\n    for x in s:\n        yield x\n",
    "async-for": "async def f(s: set[str]):\n    async for x in s:\n        yield x\n",
    "comprehension": "def f(s: set[str]):\n    return [x for x in s]\n",
    "genexp": "def f(s: set[str]):\n    return (x for x in s)\n",
    "tuple-unpack": "def f(s: set[str]):\n    a, b = s\n    return a, b\n",
    "starred": "def f(s: set[str]):\n    head, *rest = s\n    return head\n",
    "spread": "def f(s: set[str]):\n    return [*s]\n",
    "list": "def f(s: set[str]):\n    return list(s)\n",
    "tuple": "def f(s: set[str]):\n    return tuple(s)\n",
    "iter": "def f(s: set[str]):\n    return iter(s)\n",
    "next-iter": "def f(s: set[str]):\n    return next(iter(s))\n",
    "reversed": "def f(s: set[str]):\n    return reversed(s)\n",
    "enumerate": "def f(s: set[str]):\n    return list(enumerate(s))\n",
    "zip": "def f(s: set[str], t):\n    return list(zip(s, t))\n",
    "sum": "def f(s: set[float]):\n    return sum(s)\n",
    "join": 'def f(s: set[str]):\n    return ", ".join(s)\n',
    "yield-from": "def f(s: set[str]):\n    yield from s\n",
    "map": "def f(s: set[str]):\n    return list(map(str.upper, s))\n",
    "filter": "def f(s: set[str]):\n    return list(filter(None, s))\n",
    "fromkeys": "def f(s: set[str]):\n    return dict.fromkeys(s)\n",
    "pop": "def f(s: set[str]):\n    while s:\n        yield s.pop()\n",
    "str": "def f(s: set[str]):\n    return str(s)\n",
    "f-string": 'def f(s: set[str]):\n    return f"{s}"\n',
    "percent": 'def f(s: set[str]):\n    return "%s" % (s,)\n',
    "max-key": "def f(s: set[str], cost):\n    return max(s, key=cost)\n",
    "min-key": "def f(s: set[str], cost):\n    return min(s, key=cost)\n",
    "set-literal": "def f():\n    for x in {1, 2}:\n        yield x\n",
    "set-comp": "def f(n):\n    for x in {i for i in n}:\n        yield x\n",
    "set-call": "def f(n):\n    for x in set(n):\n        yield x\n",
    "frozenset-call": "def f(n):\n    for x in frozenset(n):\n        yield x\n",
    "operator": "def f(s: set[str], t):\n    for x in s ^ set(t):\n        yield x\n",
    "union": "def f(s: set[str], t):\n    for x in s.union(t):\n        yield x\n",
    "intersection": "def f(s: set[str], t):\n    for x in s.intersection(t):\n        yield x\n",
    "difference": "def f(s: set[str], t):\n    for x in s.difference(t):\n        yield x\n",
    "copy": "def f(s: set[str]):\n    t = s.copy()\n    for x in t:\n        yield x\n",
    "dict-view-operator": "def f(d, s: set[str]):\n    for x in d.keys() & s:\n        yield x\n",
    "ifexp": "def f(p, n, m):\n    for x in set(n) if p else set(m):\n        yield x\n",
    "walrus": "def f(n):\n    if s := set(n):\n        for x in s:\n            yield x\n",
    "augmented": "def f(n, t):\n    s = set(n)\n    s |= t\n    for x in s:\n        yield x\n",
    "typing-Set": "from typing import Set\n\n\ndef f(r: Set[str]):\n    return [x for x in r]\n",
    "string-annotation": 'def f(r: "set[str]"):\n    return [x for x in r]\n',
    "attribute": (
        "class C:\n"
        "    def __init__(self, n):\n"
        "        self.r = set(n)\n"
        "\n"
        "    def g(self):\n"
        "        for x in self.r:\n"
        "            yield x\n"
    ),
}

#: Every shape the pass claims *not* to report. Half of these read a set and
#: do not read its order; the rest read something that is not a set and share
#: a name or a method with one. A false positive is worse than a miss, because
#: the check's whole value is that a red gate means something.
FORMS_THAT_STAY_QUIET = {
    "sorted": "def f(s: set[str]):\n    return sorted(s)\n",
    "sorted-key": "def f(s: set[str], c):\n    return sorted(s, key=c)\n",
    "membership": "def f(s, x):\n    return x in set(s)\n",
    "len": "def f(s):\n    return len(set(s))\n",
    "add": "def f(s: set[str], x):\n    s.add(x)\n",
    "discard": "def f(s: set[str], x):\n    s.discard(x)\n",
    "mutation": "def f(s):\n    t = set(s)\n    t.add(1)\n    return t\n",
    "dict": "def f(d):\n    return [k for k in d]\n",
    "dict-view": "def f(d):\n    return [k for k in d.keys()]\n",
    "min-no-key": "def f(s: set[str]):\n    return min(s)\n",
    "max-no-key": "def f(s: set[str]):\n    return max(s)\n",
    "list-pop": "def f(rows):\n    xs = list(rows)\n    return xs.pop()\n",
    "dict-pop": "def f(d, k):\n    return d.pop(k)\n",
    "str-of-list": "def f(rows):\n    xs = list(rows)\n    return str(xs)\n",
    "rebound": (
        "def f(names):\n"
        "    s = set(names)\n"
        "    s = sorted(s)\n"
        "    for x in s:\n"
        "        yield x\n"
    ),
    "other-scope": (
        "def a(names):\n"
        "    x = set(names)\n"
        "    return sorted(x)\n"
        "\n"
        "\n"
        "def b(rows):\n"
        "    x = [1, 2, 3]\n"
        "    for i in x:\n"
        "        yield i\n"
    ),
}


#: Spellings of a real-clock read that the pass has to find. The three
#: `datetime` forms all resolve to one name, which is why the list entry is
#: written out in full rather than as the tail a caller happens to type.
CLOCK_READS_THAT_FIRE = {
    "import": "import time\n\n\ndef f():\n    return time.monotonic()\n",
    "from-import": "from time import monotonic\n\n\ndef f():\n    return monotonic()\n",
    "assigned-alias": "import time\n\nnow = time.monotonic\n\n\ndef f():\n    return now()\n",
    "nanoseconds": "import time\n\n\ndef f():\n    return time.perf_counter_ns()\n",
    "clock-gettime": "import time\n\n\ndef f():\n    return time.clock_gettime(0)\n",
    "clock-gettime-ns": "import time\n\n\ndef f():\n    return time.clock_gettime_ns(0)\n",
    "datetime-module": "import datetime\n\n\ndef f():\n    return datetime.datetime.now()\n",
    "datetime-class": "from datetime import datetime\n\n\ndef f():\n    return datetime.now()\n",
    "datetime-alias": "import datetime as dt\n\n\ndef f():\n    return dt.datetime.now()\n",
    "utcnow": "from datetime import datetime\n\n\ndef f():\n    return datetime.utcnow()\n",
    "asyncio-sleep": "import asyncio\n\n\nasync def f():\n    await asyncio.sleep(1)\n",
}

#: Calls whose name ends in a listed one and which are not it. Each is a
#: receiver the pass has no reason to think is the standard library.
CLOCK_READS_THAT_ARE_NOT = {
    "self-attribute": "class C:\n    def g(self):\n        return self.time.monotonic()\n",
    "row-attribute": "def f(row):\n    return row.datetime.now()\n",
    "imported-namespace": "import mock\n\n\ndef f():\n    return mock.time.time()\n",
    "another-package": "from mypkg import time\n\n\ndef f():\n    return time.monotonic()\n",
}

#: A real clock bound somewhere the parser does not follow and called later.
#: Every one of these reads a machine clock and none is reported.
CLOCKS_HANDED_AROUND = {
    "parameter-default": "import time\n\n\ndef f(wall_clock=time.monotonic):\n    return wall_clock()\n",
    "walrus": "import time\n\n\ndef f():\n    if c := time.monotonic:\n        return c()\n",
    "tuple-unpacking": "import time\n\n\ndef f():\n    a, b = time.monotonic, 1\n    return a()\n",
    "class-attribute": "import time\n\n\nclass C:\n    clock = time.monotonic\n\n\ndef f():\n    return C.clock()\n",
    "subscript": 'import time\n\nCLOCKS = {"m": time.monotonic}\n\n\ndef f():\n    return CLOCKS["m"]()\n',
    "partial": "import functools\nimport time\n\n\ndef f():\n    return functools.partial(time.monotonic)()\n",
}


@pytest.fixture(scope="module")
def reading_sets(tmp_path_factory):
    """The module above, written out and scanned."""
    module = tmp_path_factory.mktemp("sets") / "step.py"
    module.write_text(SIMULATED_PATH_READING_SETS, encoding="utf-8")
    return str(module), SetIterationLint().check(str(module))


class TestTheSetIterationLint:
    """The static half: a set read in order on the simulated path fails CI."""

    def test_the_simulated_path_is_clean_today(self):
        """This assertion is the gate. It is what breaks on the day one lands."""
        lint = SetIterationLint()
        code, report = lint.check(SIMULATED_PATH)
        assert code == 0
        assert report == (
            f"set-iteration lint: clean over {len(lint.modules(SIMULATED_PATH))} "
            "module(s)"
        )

    def test_it_fires_on_an_injected_read_and_names_every_one(self, reading_sets):
        module, (code, report) = reading_sets
        assert code == 1
        assert report == "\n".join(
            [
                f"set-iteration lint: {len(EXPECTED_READS)} ordered read(s) of a set:",
                *(
                    f"  {module}:{line_of(fragment)}  {form}  in {scope}"
                    for fragment, form, scope in EXPECTED_READS
                ),
                (
                    "A set hands its members back in hash order, and a string is "
                    "hashed from a seed the interpreter picks per process, so two "
                    "runs of one configuration read the same set in two orders and "
                    "diverge with nothing to point at. Sort at the point of "
                    "iteration, or decide the order once from a sorted list and "
                    "carry that: a dict keyed from sorted(s) holds an order, a "
                    "dict keyed from s holds the hash table."
                ),
            ]
        )

    @pytest.mark.parametrize("scope", BEYOND_THE_PARSER)
    def test_what_it_cannot_see_is_recorded_rather_than_claimed(
        self, reading_sets, scope
    ):
        """Each of these iterates a set, and the report does not mention it."""
        _module, (_code, report) = reading_sets
        assert f"in {scope}" not in report

    @pytest.mark.parametrize(
        "source", FORMS_THAT_FIRE.values(), ids=FORMS_THAT_FIRE.keys()
    )
    def test_every_form_it_claims_to_report_is_reported(self, source):
        """The catch-list as a regression set rather than as a paragraph."""
        assert SetIterationLint().scan_source(source, "engine.py") != ()

    @pytest.mark.parametrize(
        "source", FORMS_THAT_STAY_QUIET.values(), ids=FORMS_THAT_STAY_QUIET.keys()
    )
    def test_reading_a_set_without_reading_its_order_is_not_reported(self, source):
        """And the shapes that share a name or a method with one."""
        assert SetIterationLint().scan_source(source, "engine.py") == ()

    def test_a_name_rebound_to_the_remedy_stops_being_a_set(self):
        """The one false positive that would get the check switched off.

        A developer reads the report, sorts the members in place, and re-runs
        it. If the name still counts as a set the check fails the corrected
        code and tells the author to do what they have just done.
        """
        source = (
            "def f(names):\n"
            "    s = set(names)\n"
            "    s = sorted(s)\n"
            "    for x in s:\n"
            "        yield x\n"
        )
        assert SetIterationLint().scan_source(source, "engine.py") == ()

    def test_a_name_is_a_set_only_where_it_was_bound_one(self):
        """`x`, `s` and `ready` are short names two functions both use."""
        source = (
            "def a(names):\n"
            "    x = set(names)\n"
            "    return sorted(x)\n"
            "\n"
            "\n"
            "def b(rows):\n"
            "    x = [1, 2, 3]\n"
            "    for i in x:\n"
            "        yield i\n"
        )
        assert SetIterationLint().scan_source(source, "engine.py") == ()

    def test_an_annotated_parameter_is_enough_to_recognise_one(self):
        source = "def charge(ready: set[str]):\n    return [x for x in ready]\n"
        reads = SetIterationLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:2  comprehension over ready  in charge"
        ]

    def test_an_annotated_return_is_enough_too(self):
        """The form a caller writes when the set is built somewhere else."""
        source = (
            "def ready() -> set[str]:\n"
            "    return set()\n"
            "\n"
            "\n"
            "def charge():\n"
            "    for x in ready():\n"
            "        yield x\n"
        )
        reads = SetIterationLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:6  for-loop over ready()  in charge"
        ]

    def test_a_field_annotated_on_the_class_is_read_through_self(self):
        """A frozen dataclass has no `__init__` body to bind the name in."""
        source = (
            "class Step:\n"
            "    ready: set[str]\n"
            "\n"
            "    def charge(self):\n"
            "        for x in self.ready:\n"
            "            yield x\n"
        )
        reads = SetIterationLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:5  for-loop over self.ready  in charge"
        ]

    def test_a_set_bound_through_another_name_is_still_one(self):
        """The binding can come after the read; the pass runs to a fixed point."""
        source = (
            "def charge():\n    for x in later:\n        yield x\n\n\nlater = set()\n"
        )
        reads = SetIterationLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:2  for-loop over later  in charge"
        ]

    def test_summing_a_set_is_reported_because_float_addition_reorders(self):
        source = "def total(costs: set[float]):\n    return sum(costs)\n"
        reads = SetIterationLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:2  sum() over costs  in total"
        ]

    def test_turning_the_lint_off_makes_the_tree_claim_nothing(self):
        off = SetIterationLint(enabled=False)
        assert off.scan_source(SIMULATED_PATH_READING_SETS, "step.py") == ()
        assert off.report((), 1) == (
            "set-iteration lint: disabled, so this tree makes no claim about "
            "its set order"
        )


class TestHowFarOneAllowListEntryReaches:
    """What a path entry stands for, and the guard that keeps it to one file."""

    def test_each_entry_names_exactly_one_module_it_excuses(self):
        """The guard. An entry matching two files excuses both of them.

        The matcher compares a tail of the path, so an entry naming a short
        enough tail excuses every file ending in it, and an entry naming a file
        that has been moved or deleted excuses nothing while still reading as a
        recorded decision. Both are the same question -- how many modules does
        this entry stand for -- and one is the answer.
        """
        modules = ClockSourceLint.modules(SIMULATED_PATH)
        for entry, reason in DEFAULT_ALLOW_LIST.items():
            alone = ClockSourceLint(allow_list={entry: reason})
            matched = [path for path in modules if alone.allowed(path) is not None]
            assert len(matched) == 1, f"{entry} matches {len(matched)} module(s)"

    def test_two_files_sharing_a_whole_tail_are_both_excused(self, tmp_path):
        """The gap the guard above looks for, written down rather than assumed.

        Both paths end in the entry on a segment boundary, so both are matches
        and nothing in the report says a second file was covered. This is
        multiplicity, which the guard catches, and not anchoring, which is a
        different question answered below.
        """
        lint = ClockSourceLint(allow_list={"detect/watchdog.py": "the sampler"})
        assert lint.allowed("/one/atom/compass/detect/watchdog.py")
        assert lint.allowed("/another/vendored/detect/watchdog.py")

    def test_a_directory_merely_ending_in_the_entry_is_not_excused(self):
        """The anchoring half, which is CA-5's rule and the guard's foundation.

        Counting how many modules an entry stands for only means something if
        the match runs to a segment boundary; without that the count is taken
        over the scanned tree while the exemption reaches outside it.
        """
        lint = ClockSourceLint(allow_list={"detect/watchdog.py": "the sampler"})
        assert lint.allowed("detect/watchdog.py")
        assert lint.allowed("/one/atom/compass/NOTdetect/watchdog.py") is None


class TestWhatCountsAsAClockRead:
    """Which call is the machine's clock, and which one merely reads like it.

    The allow-list decides which *files* are excused and is settled above. This
    is the other comparison in the same pass: which *calls* are reads at all.
    It had the same unanchored shape the allow-list had -- a tail match, so any
    receiver whose attribute was spelled like a module collected the whole
    list -- and it gets the same answer, which is to compare the resolved name
    exactly against a list written out in full.
    """

    @pytest.mark.parametrize(
        "source", CLOCK_READS_THAT_FIRE.values(), ids=CLOCK_READS_THAT_FIRE.keys()
    )
    def test_every_spelling_of_a_real_read_is_found(self, source):
        assert ClockSourceLint().scan_source(source, "engine.py") != ()

    @pytest.mark.parametrize(
        "source", CLOCK_READS_THAT_ARE_NOT.values(), ids=CLOCK_READS_THAT_ARE_NOT.keys()
    )
    def test_a_name_merely_ending_in_a_listed_one_is_not_a_read(self, source):
        """A false positive here is a red gate on code that is doing nothing wrong."""
        assert ClockSourceLint().scan_source(source, "engine.py") == ()

    @pytest.mark.parametrize(
        "source", CLOCKS_HANDED_AROUND.values(), ids=CLOCKS_HANDED_AROUND.keys()
    )
    def test_a_clock_bound_where_the_parser_does_not_follow_is_not_reported(
        self, source
    ):
        """The gap recorded as measured rather than described.

        Each of these reads a machine clock at run time. Only an import and a
        plain assignment are followed, so the binding is invisible and the
        later call looks like any other. Asserted absent so the limitation is
        re-checked rather than believed.
        """
        assert ClockSourceLint().scan_source(source, "engine.py") == ()

    def test_the_three_datetime_spellings_resolve_to_one_name(self):
        """Which is why the list entry is the full name and not the tail."""
        reported = {
            str(read).split("  ")[1]
            for source in (
                "import datetime\n\n\ndef f():\n    return datetime.datetime.now()\n",
                "from datetime import datetime\n\n\ndef f():\n    return datetime.now()\n",
                "import datetime as dt\n\n\ndef f():\n    return dt.datetime.now()\n",
            )
            for read in ClockSourceLint().scan_source(source, "engine.py")
        }
        assert reported == {"datetime.datetime.now"}
