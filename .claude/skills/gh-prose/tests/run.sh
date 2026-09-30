#!/bin/sh
# bad/ must all fail, good/ must all pass. File name: <name>.<kind>.md
cd "$(dirname "$0")"; rc=0
for f in bad/*.md good/*.md; do
  k=$(basename "$f" .md); k=${k##*.}
  python3 ../scripts/lint_gh_prose.py --kind "$k" "$f" >/dev/null; got=$?
  want=1; case $f in good/*) want=0;; esac
  [ $got -eq $want ] && echo "ok   $f" || { echo "FAIL $f"; rc=1; }
done; exit $rc
