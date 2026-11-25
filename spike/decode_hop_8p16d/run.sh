#!/bin/bash
# The decode engine step's hop split (#4151), on #4029's 8P16D C1228 short cell.
#   host:          run.sh prep DIR    DIR/ATOM: #4029's measured tree with hopstamps.py and hooks.patch
#   CPU container: run.sh sim DIR/ATOM OUT [DURATION] [WARMUP_PER_LANE]: #4029's run.sh sim, stamps into OUT.frames
#   host:          python3 $TOOLS/rank.py OUT 60; python3 split.py OUT 60
# BASE (#4029's run 2 tree: a5691dd27 + #3996 + #3997 + #4004 + frames.patch) holds the c307 laws' `atom_digest`;
# hopstamps.py and both hooks live under atom/compass/, which the digest leaves out. Each stamping process writes
# <pid>.hop and <pid>.hop.json beside the frames (COMPASS_TRACE_DIR).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-/var/tmp/xingjche_simspeed/tree/m4029b/ATOM}
TOOLS=${TOOLS:-/var/tmp/xingjche_simspeed/spike4029}  # #4029's run.sh and rank.py (PR #4118)
case ${1:-} in
prep)
  DIR=${2:?DIR}
  [ -e "$DIR" ] && { echo "refused: $DIR exists"; exit 1; }
  mkdir -p "$DIR/ATOM" && tar -C "$BASE" -cf - --exclude=__pycache__ . | tar -C "$DIR/ATOM" -xf - || exit 1
  cp "$HERE/hopstamps.py" "$DIR/ATOM/atom/compass/hopstamps.py" || exit 1
  patch -s -d "$DIR/ATOM" -p1 < "$HERE/hooks.patch" || exit 1
  echo "$(cat "$BASE/MEASURED") + #4151 hopstamps $(cd "$HERE" && cat hopstamps.py hooks.patch | sha256sum | cut -c1-12)" \
    > "$DIR/ATOM/MEASURED"
  ;;
sim) shift; exec bash "$TOOLS/run.sh" sim "$@" ;;
*) sed -n '2,5p' "$0"; exit 2 ;;
esac
