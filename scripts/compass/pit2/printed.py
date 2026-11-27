# SPDX-License-Identifier: MIT
"""A calibration command `compass advice` printed, saved to a file, read for the
pit2 steps (cal_step.sh, real_step.sh).

    python3 printed.py localize CMD_FILE IP   the command, its Store on IP
    python3 printed.py out CMD_FILE           its calibration directory
    python3 printed.py prefix CMD_FILE IP     its variables, localized, as
                                              `export` arguments

CMD_FILE holds the `made by:` command: its first line that is not indented,
so the notes advice indents under it may be saved along. The prefill's offload
config names the Store of the host the search ran on
(``mooncake_store.local_hostname``); `localize` puts IP for every occurrence of
that address, the master's and the metadata server's included, as the Store a
step starts runs on its own node. A command naming no Store is unchanged.
"""

import json
import re
import shlex
import sys

HOSTNAME = re.compile(r'"mooncake_store\.local_hostname"\s*:\s*"([^"]+)"')


def command(path: str) -> str:
    """The command in the file at `path`: its first line not indented."""
    with open(path) as f:
        for line in f:
            if line.strip() and not line[0].isspace():
                return line.strip()
    raise SystemExit(f"{path}: no command in it")


def localize(cmd: str, ip: str) -> str:
    """`cmd` with the Store host it names replaced by `ip` throughout."""
    found = HOSTNAME.search(cmd)
    return cmd.replace(found.group(1), ip) if found else cmd


def split(cmd: str) -> tuple[list[str], str]:
    """`cmd`'s variable assignments before calibrate.sh, and calibrate.sh's
    first argument, the calibration's directory."""
    words = shlex.split(cmd)
    at = next((k for k, w in enumerate(words) if w.endswith("/calibrate.sh")), None)
    if at is None or at + 1 >= len(words):
        raise SystemExit(f"no calibrate.sh OUT in: {cmd[:200]}")
    return words[:at], words[at + 1]


def main(argv: list[str]) -> None:
    verb, path, *rest = argv
    cmd = command(path)
    if verb == "localize":
        print(localize(cmd, rest[0]))
    elif verb == "out":
        print(split(cmd)[1])
    elif verb == "prefix":
        words, _ = split(localize(cmd, rest[0]))
        bad = [w for w in words if "=" not in w]
        if bad:
            raise SystemExit(
                f"not a variable before calibrate.sh: {json.dumps(bad[:3])}"
            )
        print(shlex.join(words))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
