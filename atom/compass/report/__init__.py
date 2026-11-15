# SPDX-License-Identifier: MIT
"""What a run writes for a person to read once it ends, from the files in its
directory, changing none of its results.

- `split` -- each request's latency split into queue, prefill, KV transfer and
  decode, and its LMCache loads and stores.
"""
