# SPDX-License-Identifier: MIT
"""pytest plugin: append ".<pid>" to --junitxml, so a script that runs pytest
more than once with one caller's arguments (gate_gpu.sh) keeps each run's file.

    scripts/compass/gate_gpu.sh -p spike.junit_per_process --junitxml=<path>
"""

import os

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if config.option.xmlpath:
        config.option.xmlpath += f".{os.getpid()}"
