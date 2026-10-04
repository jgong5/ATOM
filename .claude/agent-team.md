---
repo: jgong5/ATOM
integration_branch: feature/atomcompass_new
gate_task: scripts/compass/gate_cpu.sh
gate_wave: scripts/compass/gate_gpu.sh
design_entry: atom/compass/design/README.md
never_touch: ROCm/ATOM
new_tests_dir: tests/compass
worktree_root: /workspace/llm_infer_deploy_study/perf_modeling/compass-worktrees
import_check: python3 -c "import atom; print(atom.__file__)"
---
Git, `gh`, `python3`, `gate_wave` and the `agent-team` scripts run in the
GPU container through `gpu_docker/shell.sh`, never on the host;
`gpu_docker/CLAUDE.md` gives the host-to-container path mapping.
`worktree_root` is a container path. Scratch and working logs go in
`agent_scratch/`, beside the repository.

**`gate_task` runs in the CPU container**, node 18's `xiaobizh_n18_cpu`, on a
`scripts/compass/snapshot.sh` tarball of the head copied in with `docker cp`
(`scripts/compass/README.md`), and is judged against the same run on the
base. Not in the GPU container: kernel tests in the CPU tier run on the
device there and fail at the base too. New tests in `new_tests_dir` are
CPU-only: they pass in that container with no driver.

**Ownership.** Git in the container runs as root and leaves files root-owned,
which makes later host-side edits fail silently. After every git command that
writes files (worktree add, fetch into a branch, merge, checkout), chown the
worktree it ran in and the shared `.git`:

```
chown -R 13797:13797 <worktree> "$(git rev-parse --git-common-dir)"
find <worktree> "$(git rev-parse --git-common-dir)" ! -uid 13797 | head -1   # prints nothing
```

**The local integration branch.** The Compass scripts (`scripts/compass/`)
diff against the local `feature/atomcompass_new` when it exists, before
`fork/feature/atomcompass_new`, so a stale local branch gives them a wrong
base unless `COMPASS_INTEGRATION_REF` names the ref. The fast-forward after
every landing keeps it current.

**Python resolution.** Run `python3` and `pytest` from the tree's root, or
with `PYTHONPATH` set to it: the container has another ATOM checkout, and
`atom` resolves there from anywhere else. A container that runs a tree
mounts `worktree_root`'s parent, not the tree alone; mounting one tree makes
`pytest` fail with a path error.

**After a container rebuild** (`/root` does not survive `teardown.sh`), run
`git config --global --add safe.directory '*'` and `gh auth setup-git` before
any git command. `setup-git` needs a `gh` login,
`/root/.config/gh/hosts.yml`, which a full teardown also discards, so `gh
auth login` or the token file `gpu_docker/CLAUDE.md` names may be needed
first (unverified: untested after a real teardown). Reinstall `gh stack`
with `./shell.sh /workspace/gpu_docker/install-gh-stack.sh`.

**`gh` version.** `pr_state.py` asks `gh pr view` for
`closingIssuesReferences`, which the image's apt `gh` (2.45.0) rejects as an
unknown field, so every PR fails. After a rebuild, run
`./shell.sh /workspace/gpu_docker/install-gh.sh`: it installs a current
release into `/usr/local/bin`, ahead of the apt copy on `PATH`.
