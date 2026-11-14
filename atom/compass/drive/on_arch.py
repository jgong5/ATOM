# SPDX-License-Identifier: MIT
"""Run a Python module in a process whose aiter answers a declared device arch.

    python -m atom.compass.drive.on_arch ARCH -m MODULE [ARGS...]

aiter is imported first, as on a device of ARCH (`driverless.import_aiter_on`);
then MODULE runs as ``python -m MODULE ARGS`` would. `serve` starts a simulated
server this way, so the server resolves its config, an FP8 quantization's dtype
among it, as on its machine's device.
"""

import runpy
import sys

from atom.compass.capture.driverless import import_aiter_on


def main(argv: list[str]) -> None:
    arch, flag, module, *args = argv
    if flag != "-m":
        raise SystemExit(f"usage: on_arch ARCH -m MODULE [ARGS...]; got {flag!r}")
    import_aiter_on(arch)
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
