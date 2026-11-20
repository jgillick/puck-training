#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS_DIR="$SCRIPT_DIR/runs"

# With no arguments, list runs and their sizes.
if [[ $# -eq 0 ]]; then
    shopt -s nullglob
    printf "%-24s %10s  %s\n" RUN SIZE STATUS
    for d in "$RUNS_DIR"/*/; do
        name=${d%/}; name=${name##*/}
        size=$(du -sh "$d" | cut -f1)
        status=""; [[ -f "$d/merged.mp4" ]] && status=cleaned
        printf "%-24s %10s  %s\n" "$name" "$size" "$status"
    done
    exit 0
fi

run_dir=${1%/}
run_dir_name=${run_dir##*/}
[[ -d "$run_dir" ]] || { echo "Error: directory not found: $run_dir" >&2; exit 1; }

cleanup_dir="${run_dir}/${run_dir_name}-cleanup"
echo "Cleaning $run_dir into $cleanup_dir"
mkdir -p "$cleanup_dir"
shopt -s nullglob

# Archive unwanted model checkpoints.
kept=0; archived=0; #counters
for f in "$run_dir"/model_*.pt; do
    num=${f##*model_}; num=${num%.pt}
    if ! [[ $num =~ ^[0-9]+$ ]]; then continue; fi # check filename format
    n=$((10#$num))
    if (( (n % 100 == 99) ||
            (n <= 2000 ? n % 200 == 0 :
            n <= 4000 ? n % 500 == 0 :
            n % 1000 == 0) )) && (( n != 0 )); then
        ((kept++))
    else
        mv -- "$f" "$cleanup_dir/"
        ((archived++))
    fi
done
(( archived + kept > 0 )) && echo "Kept $kept model checkpoints, archived $archived."

# Merge videos, then trash the originals.
if [[ -d "$run_dir/videos" ]]; then
    echo "Merging videos in $run_dir/videos into $run_dir/merged.mp4"
    ffmpeg -f concat -safe 0 \
        -i <(ls -tr $run_dir/videos/*.mp4 | sed "s|^|file '$PWD/|;s|$|'|") \
        -filter:v "setpts=0.5*PTS" -an -c:v libx264 -crf 20 -preset fast \
        -pix_fmt yuv420p "$run_dir/merged.mp4"
    mv "$run_dir/videos" "$cleanup_dir/"
fi

if [[ -z "$(ls -A "$cleanup_dir" 2>/dev/null)" ]]; then
    rm -rf "$cleanup_dir"; # nothing to trash
elif command -v trash &> /dev/null; then # run 'trash' if available
    trash "$cleanup_dir"
    echo "Done! $cleanup_dir moved to trash."
else
    echo "Done! $cleanup_dir left in place (trash command not found)."
fi
