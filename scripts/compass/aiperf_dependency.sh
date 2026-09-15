#!/usr/bin/env bash
# Apply or verify the reviewed fix in a private AIPerf checkout.
set -euo pipefail

base=0d2aa0572ac685943d38c580675c4a61023581d3
patch_sha=4d9410f3bab7862f17bc14200f0414c7de82a6391e6601491a0d10881b34b796
patch_file="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/patches/aiperf-zero-warmup.patch"
files=(
    src/aiperf/common/models/credit_models.py
    src/aiperf/timing/config.py
    src/aiperf/timing/phase/runner.py
)
before=(
    2ecfe10b926e5b2dee82cb9fa7dde1128b333334ddd9e4719f98117492ae9877
    07c4a4e01621be843aba171c1ba8c06d872498c892ebc2017d2153846563b61f
    abb24030929ee385a312b67ea886e8567c5aff732b1055c54c661826046c8ad1
)
after=(
    cbce0016b176e8110540fe6742fe92246794714a05e3ed92cc222f72655fc995
    4c7fef26153cd438b297baf52158bb17c08bbfac8cd391dd3de7ec6421debbea
    47c7853430ba262cfe4967c8cc7d3997ab23f2f57b2aa76412da7f6fcbb65fa3
)

die() { printf 'AIPerf dependency refused: %s\n' "$*" >&2; exit 2; }
check_hash() {
    local actual
    [[ -f "$1" && ! -L "$1" ]] || die "missing or symlinked file: $1"
    actual=$(sha256sum -- "$1")
    [[ ${actual%% *} == "$2" ]] || die "unexpected bytes: $1"
}
check_files() {
    local i=0 expected
    for expected in "$@"; do
        check_hash "$source_root/${files[$i]}" "$expected"
        i=$((i + 1))
    done
}

[[ $# == 2 ]] || die "usage: $0 {check-base|apply|verify} PRIVATE_AIPERF_CHECKOUT"
action=$1
case "$action" in check-base|apply|verify) ;; *) die "unknown action: $action" ;; esac
source_root=$(cd -- "$2" && pwd -P)
git_cmd=(git -c "safe.directory=$source_root" -C "$source_root")
[[ $("${git_cmd[@]}" rev-parse --show-toplevel) == "$source_root" ]] || die "name the checkout root"
[[ $("${git_cmd[@]}" rev-parse HEAD) == "$base" ]] || die "HEAD must be $base"
check_hash "$patch_file" "$patch_sha"

if [[ $action == check-base || $action == apply ]]; then
    [[ -z $("${git_cmd[@]}" status --porcelain=v1 --untracked-files=all) ]] || die "base checkout has changes"
    check_files "${before[@]}"
    if [[ $action == check-base ]]; then
        printf 'dependency=aiperf state=pristine base_commit=%s\n' "$base"
        exit 0
    fi
    "${git_cmd[@]}" apply --check --whitespace=error-all "$patch_file"
    "${git_cmd[@]}" apply --whitespace=error-all "$patch_file"
fi

check_files "${after[@]}"
expected_status=$(printf ' M %s\n' "${files[@]}" | LC_ALL=C sort)
actual_status=$("${git_cmd[@]}" status --porcelain=v1 --untracked-files=all | LC_ALL=C sort)
[[ $actual_status == "$expected_status" ]] || die "checkout has changes beyond the unstaged reviewed patch"
printf 'dependency=aiperf state=atomcompass-zero-warmup-v1 base_commit=%s patch_sha256=%s\n' "$base" "$patch_sha"
