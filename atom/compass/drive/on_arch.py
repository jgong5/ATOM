# SPDX-License-Identifier: MIT
"""Run a Python program in a process whose aiter answers a declared device arch.

    python -m atom.compass.drive.on_arch ARCH -m MODULE [ARGS...]
    python -m atom.compass.drive.on_arch ARCH SCRIPT [ARGS...]

aiter is imported first, as on a device of ARCH (`driverless.import_aiter_on`);
then the program runs as ``python -m MODULE ARGS`` or ``python SCRIPT ARGS``
would. `serve` starts a simulated server this way, so the server resolves its
config, an FP8 quantization's dtype among it, as on its machine's device.
"""

import runpy
import sys

from atom.compass.capture.driverless import import_aiter_on


def main(argv: list[str]) -> None:
    arch, *program = argv
    import_aiter_on(arch)
    if program[0] == "-m":
        sys.argv = program[1:]
        runpy.run_module(program[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = program
        runpy.run_path(program[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
