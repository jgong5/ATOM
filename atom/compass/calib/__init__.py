# SPDX-License-Identifier: MIT
"""The unified calibration procedure v1.1 (wuhuikx/LLM-Booster#739 sections 2-4):
a step-law calibration workload designed from a deployment's configuration and
the law's columns, independent of any target workload and of concurrency.

A design is a config (`scripts/compass/calib/*.json`, read by `deploy.load`):
the deployment (data-parallel ranks, capture rungs, scheduler bounds, KV
capacity per rank, speculative drafts and acceptance, prefix-cache unit), the
law's columns per vector (`columns`), the hardware noise and the acceptance
criteria frozen before a record, and the segments, each a kind `design`
generates from the deployment.

- `deploy`: the config, and the reachable range of each vector's drivers.
- `columns`: a step's value of each named column: the step linear backend's
  own counts for every column it prices, and a provisional definition for an
  opt-in column this tree's backend does not price yet.
- `design`: the segments (`gen_v1.py`): traces as `client.sh --traces` sends
  them, each lane's pin, and the common departure schedule of the drains.
- `engine`: a CPU rehearsal of a design on a law (`dry_v1.sh`): the engine's
  steps as a simplified DP scheduler forms them, priced by the law and the host
  law, each marked GPU-bound or not, and the predicted GPU minutes
  (`plan_v1.py`).
- `ident`: identifiability per vector, over the GPU-bound steps only: rank,
  VIF, the worst relative prediction error G over the reachable range, and
  the levels each column takes.
- `check`: after a record, each segment's actual combinations against the
  plan's, and the same identifiability on the record's own steps (`check_v1.py`).
"""
