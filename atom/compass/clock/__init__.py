# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where simulated time is kept, and what has to stop waiting for it.

Today this holds the classified inventory of every call on ATOM's serving path
that can stop a thread: ``sync_scan.py`` finds them and ``sync_sites.json``
says what each one is.
"""
