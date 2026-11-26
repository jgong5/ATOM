# SPDX-License-Identifier: MIT
"""What a run writes for a person to read once it ends, from the files in its
directory, changing none of its results.

- `split` -- each request's latency split into queue, prefill, KV transfer and
  decode, and its LMCache loads and stores.
- `ops` -- an op-level run's time per step phase by op family, from its
  prices.
- `html` -- the run's report, one static page: the run, its headline metrics,
  a simulated run's trust, the split and the op-family split.
- `extrapolation` -- how much of a simulated run its cost sources priced past
  their support domains, on each DP group step's critical rank apart from
  every rank, off its step records: in the run summary, and over any window
  from the command line.
"""
