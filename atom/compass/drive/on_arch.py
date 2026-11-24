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
as the first step of its spawn command line: before it imports its parent's
``__main__``, MODULE or a module MODULE runs in its place.
"""

import runpy
import sys
from multiprocessing import spawn
from pathlib import Path

import torch

import atom
from atom.compass.capture.driverless import answer_as, import_aiter_on

#: Where a spawned process imports this `atom` from, before it takes its
#: parent's `sys.path`.
TREE = str(Path(atom.__file__).resolve().parents[1])


def declare(arch: str) -> None:
    """aiter, with no driver here `torch.cuda`, and every process this one
    spawns, as on a device of `arch`."""
    import_aiter_on(arch)
    if not torch.cuda.is_available():
        answer_as(arch)
    line = spawn.get_command_line

    def get_command_line(**kwds) -> list[str]:
        argv = line(**kwds)
        i = argv.index("-c") + 1
        argv[i] = (
            f"import sys; sys.path.insert(0, {TREE!r}); "
            f"from atom.compass.drive.on_arch import declare; declare({arch!r}); "
            f"{argv[i]}"
        )
        return argv

    spawn.get_command_line = get_command_line


def main(argv: list[str]) -> None:
    arch, flag, module, *args = argv
    if flag != "-m":
        raise SystemExit(f"usage: on_arch ARCH -m MODULE [ARGS...]; got {flag!r}")
    declare(arch)
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
