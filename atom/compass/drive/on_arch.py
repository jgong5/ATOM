# SPDX-License-Identifier: MIT
"""Run a Python module in a process whose aiter answers a declared device arch.

    python -m atom.compass.drive.on_arch ARCH -m MODULE [ARGS...]

aiter is imported first, as on a device of ARCH (`driverless.import_aiter_on`);
then MODULE runs as ``python -m MODULE ARGS`` would. `serve` starts a simulated
server this way, so the server resolves its config, an FP8 quantization's dtype
among it, as on its machine's device. On a host with no driver, `torch.cuda`'s
device queries answer as that device too (`driverless.answer_as`): a simulated
runner asks them as it builds, its model and its attention metadata builder.

Every process the server spawns, and each one those spawn, declares ARCH too,
before it unpickles anything that imports aiter: this module stays the
server's ``__main__``, which a spawned process imports first, as
``__mp_main__``, and the arch is in its environment (`ARCH_ENV`).
"""

import os
import runpy
import sys

import torch

from atom.compass.capture.driverless import answer_as, import_aiter_on

#: The arch a process started here declared, which the processes it spawns read.
ARCH_ENV = "ATOM_COMPASS_ON_ARCH"


def declare(arch: str) -> None:
    """aiter, and with no driver here `torch.cuda`, as on a device of `arch`."""
    import_aiter_on(arch)
    if not torch.cuda.is_available():
        answer_as(arch)


def main(argv: list[str]) -> None:
    arch, flag, module, *args = argv
    if flag != "-m":
        raise SystemExit(f"usage: on_arch ARCH -m MODULE [ARGS...]; got {flag!r}")
    declare(arch)
    os.environ[ARCH_ENV] = arch
    sys.argv = [module, *args]
    # Not `alter_sys`, which would make MODULE the `__main__` a spawned process
    # imports, and so import aiter there before anything declares the arch.
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
elif __name__ == "__mp_main__":
    declare(os.environ[ARCH_ENV])
