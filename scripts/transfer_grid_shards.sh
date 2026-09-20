#!/usr/bin/env bash
# Run the transfer grid as several shards at once, the seeds dealt round-robin over the slots, one
# process per slot: a slot is one accelerator, or one of several processes on an accelerator that
# a small model leaves idle. Every shard stores into its own directory and publishes every cell as
# it lands, so a dropped session loses at most one cell per shard, and the shards are read back as
# one curve by the curve report. Portable bash: it also runs on the notebook platforms.
#
#   scripts/transfer_grid_shards.sh --devices cuda:0,cuda:1 --per-device 2 \
#       --seeds "1 2 3 4 5" --out grid -- --task turbofan-fd001 --weights KEY SUM --manifest KEY SUM
#
#   --devices     comma-separated devices, one process group each (default: mps)
#   --per-device  processes per device (default: 1)
#   --seeds       seeds to deal out (default: "1 2 3 4 5")
#   --out         directory the shards are stored under, as <out>/shard-<n>
#   --runner      how the report is invoked (default: "uv run --env-file .env.r2")
#   --no-publish  keep the shards local
#   DRY=1         print the commands instead of running them
# Everything after `--` is passed to scripts/transfer_modes_report.py unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICES=mps
PER_DEVICE=1
SEEDS="1 2 3 4 5"
OUT=""
RUNNER="uv run --env-file .env.r2"
PUBLISH=--publish
while [ $# -gt 0 ]; do
  case "$1" in
    --devices) DEVICES=$2; shift 2 ;;
    --per-device) PER_DEVICE=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --runner) RUNNER=$2; shift 2 ;;
    --no-publish) PUBLISH=""; shift ;;
    --) shift; break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }

# The slots: device repeated per-device times, in order.
slots=()
old_ifs=$IFS; IFS=,
for device in $DEVICES; do
  IFS=$old_ifs
  n=0
  while [ "$n" -lt "$PER_DEVICE" ]; do slots+=("$device"); n=$((n + 1)); done
  IFS=,
done
IFS=$old_ifs

# Deal the seeds round-robin over the slots; a slot without a seed does not start.
count=${#slots[@]}
seed_flags=()
i=0
while [ "$i" -lt "$count" ]; do seed_flags+=(""); i=$((i + 1)); done
i=0
for seed in $SEEDS; do
  slot=$((i % count))
  seed_flags[$slot]="${seed_flags[$slot]} --seed $seed"
  i=$((i + 1))
done

pids=()
i=0
while [ "$i" -lt "$count" ]; do
  if [ -n "${seed_flags[$i]}" ]; then
    echo "== shard-$i on ${slots[$i]}:${seed_flags[$i]}"
    # shellcheck disable=SC2086  # RUNNER, PUBLISH and the seed flags are meant to split
    ${DRY:+echo} $RUNNER scripts/transfer_modes_report.py "$@" \
      --device "${slots[$i]}" --out "$OUT/shard-$i" $PUBLISH ${seed_flags[$i]} 2>&1 \
      | sed -u "s/^/[shard-$i] /" &
    pids+=($!)
  fi
  i=$((i + 1))
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done
if [ "$failed" -gt 0 ]; then
  echo "== $failed shard(s) failed; the cells stored so far are in $OUT/shard-*" >&2
  exit 1
fi
echo "== every shard done; read them with: uv run scripts/label_curve_report.py $OUT/shard-*"
