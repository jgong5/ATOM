# SPDX-License-Identifier: MIT
"""Validation of Compass's predictions against real runs.

- `cost` -- a law or a step grid priced on a real run's own steps, its error
  per step.
- `runs` -- a simulated cell graded against real runs of it, per metric.
- `bound` -- the analytical price at most the measured time on every step of
  real runs, at step and op level.
"""
