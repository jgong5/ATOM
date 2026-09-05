# Pick the least-occupied GPUs. The box is shared with about twenty containers
# and neighbours appear mid-session: two runs died with available_for_kv
# negative because someone else held 170 GB. Raising --gpu-memory-utilization
# does not help -- non_torch is charged against the budget regardless -- so the
# only fix is to land somewhere free.
pick_gpus() {  # $1 = how many
  local want=${1:-1}
  rocm-smi --showmemuse 2>/dev/null \
    | sed -n 's/^GPU\[\([0-9]\+\)\].*VRAM%): \([0-9]\+\).*/\1 \2/p' \
    | sort -k2 -n | head -"$want" | cut -d' ' -f1 | paste -sd,
}
