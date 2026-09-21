# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Static audits of ATOM's source, and the answers they are checked against.

These run over the tree, not in a simulated run: they parse ATOM's modules and
compare what they find against a checked-in list, so that a change to ATOM
fails a test rather than quietly invalidating an answer. Today that is the
inventory of blocking calls on the serving path -- `sync_scan.py` finds the
call sites, `sync_sites.json` says what each one is, and `README.md` explains
both.

Deliberately not under `atom/compass/clock/`: that package is held to a small
standard-library allowlist because it has to be constructible anywhere, and a
tool that parses files and walks directories cannot meet it. What is audited
here is about time; the auditing is not part of the clock.
"""
