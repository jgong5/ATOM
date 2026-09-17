# Private AIPerf dependency correction

`aiperf-zero-warmup.patch` corrects a lifecycle defect in AIPerf commit
`0d2aa0572ac685943d38c580675c4a61023581d3`. A trajectory containing only a future
background child can require zero priming credits. The original runner keeps
its positive placeholder target and waits for an event that no credit can
produce. It also catches unexpected sending-wait failures without propagating
them, which can make a failed phase appear to complete.

The corrected private dependency represents the derived empty warmup as zero,
waits for the empty strategy to finish normally, and completes through the
existing phase lifecycle. It propagates unexpected wait errors. Public request
counts remain positive; zero is admitted only for an AgenticReplay warmup without
an accelerated cache-pressure stage. The stats schema reports the true zero.
Branch dispatch, foreground join timing, tree accounting and scenario cutoff
semantics are unchanged, including the existing completion-order join asymmetry.

The corrected identity is **base commit plus patch SHA-256**:

```text
base_commit=0d2aa0572ac685943d38c580675c4a61023581d3
patch_sha256=4d9410f3bab7862f17bc14200f0414c7de82a6391e6601491a0d10881b34b796
dependency_state=atomcompass-zero-warmup-v1
```

Historical unpatched runs retain their original identity. The pristine AIPerf
exporters still require an unmodified checkout and intentionally refuse this
private correction. Do not relabel historical evidence or replace an active
dependency checkout to apply it.

## Apply and verify

Use a dedicated checkout at the exact base commit. Run Git and the shell helper
inside the configured development container; in the shared workspace, invoke
them through `gpu_docker/shell.sh`.

```bash
git -C "$AIPERF_EXISTING_CHECKOUT" worktree add --detach "$AIPERF_PRIVATE_CHECKOUT" \
  0d2aa0572ac685943d38c580675c4a61023581d3
bash "$ATOM_CHECKOUT/scripts/compass/aiperf_dependency.sh" check-base "$AIPERF_PRIVATE_CHECKOUT"
bash "$ATOM_CHECKOUT/scripts/compass/aiperf_dependency.sh" apply "$AIPERF_PRIVATE_CHECKOUT"
bash "$ATOM_CHECKOUT/scripts/compass/aiperf_dependency.sh" verify "$AIPERF_PRIVATE_CHECKOUT"
```

The helper pins the patch and every before/after file digest. Application
requires a clean checkout; a wrong HEAD, partial patch or unrelated edit refuses
before application. Verification requires the exact unstaged three-file patch
and refuses unrelated tracked or untracked changes. It does not install
packages, change imports, commit the vendor tree or write a success receipt.

## Portable regression

In a device-free Python environment with the AIPerf dependency installed, select
the private source explicitly:

```bash
PYTHONPATH="$AIPERF_PRIVATE_CHECKOUT/src" python -m pytest \
  "$ATOM_CHECKOUT/tests/compass/test_aiperf_zero_warmup.py" -q -p no:cacheprovider
```

The fixture uses real AIPerf metadata and `PhaseRunner`/`PhaseOrchestrator` for a
finished root with a future background child. It needs no tokenizer, GPU, trace
download or ignored result artifact. The pristine dependency fails its empty
transition and wait-error checks; the corrected dependency must complete with
zero counts, preserve the child, and propagate strategy failure, cancellation
and unexpected wait failures. Missing AIPerf skips this optional dependency
test; an installed uncorrected dependency is not treated as a pass.

The separate C1/seed-42 scenario witness exercised the actual loader, empty
warmup handoff, first-turn cache-bust payloads and a 900-second virtual window.
It observed 235 credits: 33 whole root cycles, one initial rootless child and a
final partial root cut off normally. That external evidence is not bundled
here and is not proof of nonempty cache priming, actual model performance or
paired E2E acceptance.

## Controlled clocks and raw transport boundaries

The additive `aiperf-controlled-clock.patch` scopes semantic clocks for the
controlled runner while retaining native clock defaults. After applying the
zero-warmup prerequisite, use `apply-controlled` and `verify-controlled`.

The separately pinned `aiperf-raw-transport-timestamps.patch` adds optional
`end_perf_ns` and `recv_start_perf_ns` fields to raw exports, copied directly
from the existing `RequestRecord`. The last SSE packet and
`metadata.request_end_ns` keep their existing meanings. Missing new fields in
an older export mean the transport boundaries are unavailable.

```bash
bash "$ATOM_CHECKOUT/scripts/compass/aiperf_dependency.sh" apply-controlled "$AIPERF_PRIVATE_CHECKOUT"
bash "$ATOM_CHECKOUT/scripts/compass/aiperf_dependency.sh" verify-controlled "$AIPERF_PRIVATE_CHECKOUT"
bash "$ATOM_CHECKOUT/scripts/compass/aiperf_dependency.sh" apply-export "$AIPERF_PRIVATE_CHECKOUT"
bash "$ATOM_CHECKOUT/scripts/compass/aiperf_dependency.sh" verify-export "$AIPERF_PRIVATE_CHECKOUT"
```

The export extension requires controlled-clock patch SHA-256
`c8035a3cac1061a817e87af31ab440508d9b23dfe5a3b2cc27e5708e96e45d06`.
Its patch SHA-256 is
`f2ab352de580a250ea2ea151541e92b2a741ee20603b9f66d47506642b838f3b`;
the adjacent JSON manifest pins the complete resulting dependency state.
It changes export evidence only; HTTP, SSE, clocks and credit control remain
unchanged. `tests/compass/test_aiperf_raw_timestamps.py` checks the real writer's
serialization round-trip with distinct last-SSE and transport-completion times.
