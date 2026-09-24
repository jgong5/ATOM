<!-- SPDX-License-Identifier: MIT -->
# Blocking calls on ATOM's serving path

A simulated run substitutes a predicted duration for the work a batch does. Any
call that parks a thread on the **real** clock therefore has to be accounted
for: some of them are the durations being predicted, some only need the process
to say it is idle, some are bounds that would fire on a healthy run, and most
are invisible to the result and are left exactly as they are.

Two files:

| File | What |
|---|---|
| `sync_scan.py` | finds the call sites. It reports candidates and judges none of them. |
| `sync_sites.json` | says what each one is: category, who it is waiting for, and one line of why. |

`tests/compass/test_sync_inventory.py` asserts the two agree. **A blocking call
added to ATOM later fails that test** rather than being missed, and the failure
message says which of the three things went wrong: an unclassified site, a
recorded line that moved, or a pinned line of text that is gone.

## The categories

| | Meaning | What is done to it |
|---|---|---|
| **A** | simulated time decides this, not the real clock | the duration is predicted, the reading comes from the simulated clock, or an idle loop jumps to the next event instead of spinning |
| **B** | parks until another **process** sends something, with no bound | the call is left alone; the process declares itself idle around it |
| **C1** | a bound that exists to declare something broken | raise it, or switch it off |
| **C2** | a bound that sets a cadence | keep the cadence on simulated time, poll briefly for real so the thread stays responsive |
| **ignore** | invisible to the result | leave it |
| **undecided** | the reading depends on a decision not yet made | decide it before building on the row |

**The boundary between B and C1/C2 is the bound, not the peer.** An unbounded
wait for another process is B; the same wait with a finite bound is C1 or C2 by
what the bound is for. **The boundary between B and `ignore` is the process.** A
thread parked on a queue that its own process fills does not make that process
idle — its step loop is still running — so declaring it blocked there would be
wrong, not merely redundant.

## The counts

Measured at the commit this file was written against, from the rows below:
**194 call sites plus 18 pinned points that are not a call.**

| Category | Count | Where the weight is |
|---|---|---|
| A | 23 | five forward passes, three transfer-completion polls, the idle-rank batch, five tokenizer hand-offs, three idle step loops that spin, and six clock readings the result reports |
| B | 36 | the worker RPC both ways, the nine out-of-band control commands, the front-end and engine socket threads, each request's own wait for its next chunk on all four streaming endpoints, and the two channels of the intra-device split |
| C1 | 11 | four bounded receives and queue reads, five router and server bounds, a keep-alive frame and a silence warning |
| C2 | 3 | the idle transfer drain, and the two pipeline-stage polls |
| ignore | 137 | below the replaced forward pass, inside one process, at startup, at shutdown, or in a transfer backend the simulator substitutes |
| undecided | 2 | see below |

Counting each site once, in this order, the 134 ignored **call sites** are: the
runner module the seam replaces (21), the two real RDMA transfer backends (17),
the send half of a cross-process message (24), startup and shutdown (35),
collectives inside the real forward pass (4), text scanners whose loops park on
nothing (4), and 29 others that each carry their own reason — an unstarted
thread, an in-process hand-off, a lockstep reduction inside one rank group, a
death detector, and so on.

### The two undecided rows

- **Committing a pipeline-stage send** (`atom/distributed/pp_comm.py`). The stage
  loop still asks the runner to flush pending sends, and the replacement runner
  has to answer; whether it has anything to commit is a property of that runner
  and is not settled here.
- **Multimodal input preparation** (`atom/entrypoints/openai/api_server.py`). It
  shares the thread pool with tokenization, whose service time is charged from
  the machine spec, but no service time is defined for image work.

Both are in the list rather than forced into a category, because behaviour built
from a forced row is wrong in a way that does not announce itself.

## What the scanner does, and where it can be wrong

It parses every file under `SCANNED_ROOTS` and reports calls whose **shape** can
park a thread: sleeps, socket receives, pollers, queue operations, joins, lock
and event waits, collectives, device synchronization, sends, thread-pool
hand-offs, and the worker RPC. It also reports `while` loops that go around
without calling anything that parks. Shapes match on the call's own text and
argument form, never on what a name is bound to at runtime.

Where a name alone is ambiguous, the argument form settles it, because only the
blocking reading can take that form: `d.get(key)` always passes the key
positionally and `q.get()` never does; `",".join(parts)` always passes one
iterable and `t.join(5)` a lone number. Those rules have their own tests.

**Roots are directories, not a list of files.** A file allowlist declares its
blind spot at a finer grain than it excludes: a module added beside a scanned
one is invisible while the exclusion list still reads complete. Scanning the
directory makes a new file a test failure on the day it lands, and a test
asserts no scanned file is also declared unscanned.

It over-reports on purpose. A `.send()` that never blocks in practice is still
listed, because that is where a message leaves one process for another. The
classification is what removes it from consideration, and an `ignore` row with
its reason is a better record than a silent omission.

It can be wrong in two directions, and both are visible rather than hidden:

- **Toward noise.** Some listed calls cannot park — an `enqueue` that appends to
  a thread-local buffer, a `result()` on a task already finished, a `while` loop
  walking a string. They carry an `ignore` row saying so.
- **Toward silence.** A blocking call reached through a name the shapes do not
  know, or in a directory outside `SCANNED_ROOTS`, is invisible.
  `UNSCANNED_ROOTS` names each excluded directory and why, so the boundary is
  arguable rather than implicit, and a test asserts every named root exists.

## How a row keeps naming the same call

A row's identity is `file::symbol::expression::shape#ordinal` — **not** the
line, so an edit elsewhere in a file updates a line rather than re-opening a
classification, and **with** the arguments, so two calls to one method in one
function are told apart. An earlier version keyed on the callee alone: inserting
one `call_func("flush_pp_send", ...)` above a `call_func("forward", ...)` then
silently moved every later row onto the wrong site and reported only that a line
had moved. Two sites that still share an ordinal are the same call written the
same way in one function, and a test asserts the inventory answers both the same
way, so the ordinal never carries meaning of its own.

## What the rows do not cover

Every reading of the wall clock, as opposed to every wait. The rows under
`anchors` carry the readings that gate scheduling or that the result reports —
the delay gate, the queue-age guard, and the arrival, first-token and finish
stamps — because a category already applies to them. The rest of this path's
clock reads are logging and metrics, and separating those from business logic
across the whole tree is its own pass.
