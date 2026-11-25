# SPDX-License-Identifier: MIT
"""The unified calibration procedure v1.1 (wuhuikx/LLM-Booster#739): a
step-law calibration workload designed from a deployment's configuration and
the law's columns, independent of any target workload and of concurrency.

A design is a config (`deploy.load`): the deployment (data-parallel ranks,
capture rungs, scheduler bounds, KV capacity per rank, speculative drafts and
acceptance, prefix-cache unit), the law's columns per vector (`columns`), the
hardware noise and the acceptance criteria frozen before a record, and the
segments, each a kind with its parameters.

- `deploy`: the config, read from a file or a record's deployment, and the
  sizes a segment states relative to the deployment (`deploy.count`,
  `deploy.tokens`).
- `columns`: a step's value of each named column: the step linear backend's
  own counts for every column it prices, and a provisional definition for an
  opt-in column this tree's backend does not price yet.
- `design`: the segments, generated from the deployment: the weka traces each
  lane sends, each lane's pin, and the departure schedule every drain shares
  (`scripts/compass/gen_v1.py`).
- `engine`: a CPU rehearsal of a design on a law (`dry_v1.sh`): the engine's
  steps as a simplified DP scheduler forms them, priced by the law and the host
  law, each marked GPU-bound or not, and the predicted GPU minutes
  (`plan_v1.py`).
- `ident`: identifiability per vector, over the GPU-bound steps only: rank,
  VIF, the worst relative prediction error G over the reachable range, and
  the levels each column takes.
- `check`: after a record, each segment's actual combinations against the
  plan's, and the same identifiability on the record's own steps
  (`check_v1.py`).
- `report`: the tables `plan_v1.py` and `check_v1.py` print.
"""
