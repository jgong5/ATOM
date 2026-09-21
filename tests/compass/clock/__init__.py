# SPDX-License-Identifier: MIT
"""Synthetic participants driving the clock, without ATOM and without a GPU.

Three modules and one test file:

* `deployments` — which participants a deployment has, and the floor on every
  ordered pair between them. This is where the collapse lives: adding tensor- or
  data-parallel width multiplies processes and GPUs and creates no participant.
* `participants` — what each one does between grants. Every behaviour here is a
  row of the blocking-call inventory in `atom/compass/audit/sync_sites.json`,
  named in the docstring of the method that models it.
* `harness` — the driver. It decides when a participant asks for time, which is
  the thing that decides whether a run skips idle or crawls, and it holds the
  checks that a run is not quietly wrong.
* `test_synthetic_deployments` — the six deployments, run.
"""
