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
`docker` group. `rocm-smi`, `rocminfo` and `preflight.sh` run on the host too.
The image is the node-18 tag QUICKSTART's Environment names, and its `docker
run` needs nothing more here. Other users' containers run here: touch none.

## Allocation

None: any user may start on any GPU. A GPU is free while the GPU check reads it
idle, and a run on it holds it only while it runs. Ask the developer which GPUs
their team uses here and keep to those.

## Inputs

The model repo and traces directory QUICKSTART's Environment names for node 18,
mounted read-only where it says. Both are world-readable.

## Disk

- Data directory: a fresh subdirectory per session, such as
  `/data/<user>/compass/onboard-<date>`. `/data/<user>/compass` itself may
  hold inputs and other work.
- When `/data` has no room for it: QUICKSTART's Disk fallback on the host's
  tmpfs, again a fresh subdirectory per session, such as
  `/tmp/<user>_compass/onboard-<date>`.
