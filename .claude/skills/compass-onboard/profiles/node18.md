---
hosts: hjbog-srdc-18, hjbog-srdc-18.amd.com
machine_spec: scripts/compass/machines/mi308x-hjbog18.json
gfx: gfx942
did: 0x74a2
---

# Node 18

Eight MI308X on one host, shared by several users, with no scheduler.

## Runtime

Docker on the host, run as the developer's own user, who must be in the
`docker` group. `rocm-smi`, `rocminfo` and the tree's `preflight.sh` run on the
host too. The image is the node-18 tag QUICKSTART's Environment names. Other
users' containers run here: touch none.

## Allocation

None: any user may start on any GPU. A GPU is free while the GPU check reads it
idle, and a run on it holds it only while it runs. Ask the developer which GPUs
their team uses here and keep to those.

## Inputs

The model repo and traces directory QUICKSTART's Environment names for node 18,
mounted read-only where it says. Both are world-readable. The developer builds
their own client venv (QUICKSTART's recipe) in the data directory.

## Disk

- Data directory: `/data/<user>/compass`.
- Docker's storage is `/data/docker`, on the same disk, so the container's
  writable layer takes room from it too. `/data` is ext4: when it reads full
  for users, the daemon still writes into the root-only reserve every user
  shares.
- The root disk is small: nothing goes there.
- Fallback when `/data` has no room for the developer: the data directory on
  the host's tmpfs, such as `/tmp/<user>_compass`, mounted in the container at
  the same path, and the container started with a tmpfs at each directory the
  runs fill: `--tmpfs /root/.cache:size=64g --tmpfs /root/.aiter:size=64g
  --tmpfs /root/.triton:size=16g --tmpfs /tmp:size=64g`. A tmpfs is memory: it
  is lost at reboot, so copy out what should stay.
- The writable layer then takes aiter's built modules and aiperf's bytecode:
  20 MB through step 0 of a node-18 run on 2026-10-20, when `/root/.aiter`, now
  on tmpfs, held 227 MB. Step 0 takes the container's builds; a later record
  that builds one is refused by grading.
