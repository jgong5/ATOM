# SPDX-License-Identifier: MIT
"""Device-free model capture: build ATOM's real model classes without a GPU,
run one forward under `FakeTensorMode`, and record the operators it dispatches.
"""
