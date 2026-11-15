# SPDX-License-Identifier: MIT
"""What a run writes for a person to read once it ends, from the files in its
directory, changing none of its results.

- `split` -- each request's latency split into queue, prefill, KV transfer and
  decode, and its LMCache loads and stores.
- `html` -- the run's report, one static page: the run, its headline metrics,
  a simulated run's trust and the split.
"""
