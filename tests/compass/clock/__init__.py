# SPDX-License-Identifier: MIT
"""Synthetic participants driving the clock, without ATOM and without a GPU.

What a clock-driven run is built out of:

* `deployments` — which participants a deployment has, and the floor on every
  ordered pair between them. This is where the collapse lives: adding tensor- or
  data-parallel width multiplies processes and GPUs and creates no participant.
* `participants` — what each one does between grants. Every behaviour here is a
  row of the blocking-call inventory in `atom/compass/audit/sync_sites.json`,
  named in the docstring of the method that models it.
* `harness` — the driver. It decides when a participant asks for time, which is
  the thing that decides whether a run skips idle or crawls, and it holds the
  checks that a run is not quietly wrong.

That is a description of the parts, **not a census of the directory**, and it
deliberately gives no count. A docstring that enumerates its own package is
wrong the first time anybody adds a file and then stays wrong quietly: this one
said "three modules and one test file" and was overtaken twice in one afternoon,
by two different branches, neither of which could correct it without colliding
with a review in flight. Anything else that lands beside these is found by
listing the directory.
"""
