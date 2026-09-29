#!/usr/bin/env bash
# Train, render and evaluate the "Ours" rows of Table 1.
# See docs/REPRODUCE.md for where each setting comes from.
#
#   DATA=/path/to/data ./scripts/train_table1.sh            # all rows
#   DATA=/path/to/data ./scripts/train_table1.sh Horse      # rows of one scene
set -euo pipefail

: "${DATA:?set DATA to the directory that holds colmap/ sp-sg/ defree_sfm/}"
OUT=${OUT:-output}
ONLY=${1:-}

run() {  # run <sfm> <scene> [train.py options...]
    local sfm=$1 scene=$2; shift 2
    [[ -n "$ONLY" && "$ONLY" != "$scene" ]] && return 0
    local model="$OUT/$sfm/$scene"
    python train.py  -s "$DATA/$sfm/$scene" -m "$model" --eval "$@"
    python render.py -m "$model" --skip_train
    python metrics.py -m "$model"
}

# Tanks and Temples
run colmap     Train --densify_until_iter 10000 --size_threshold_from_iter 8000
run sp-sg      Train                                                               # note 1
run defree_sfm Train --densify_until_iter 10000 --size_threshold_from_iter 8000
run colmap     Horse --densify_until_iter 20000 --size_threshold_from_iter -1      # note 2
run sp-sg      Horse --densify_until_iter 20000 --size_threshold_from_iter -1
run defree_sfm Horse --densify_until_iter 15000 --size_threshold_from_iter -1

# In-house scenes (not released)
run sp-sg      ladybug6   --densify_until_iter 10000 --size_threshold_from_iter 10000
run defree_sfm ladybug6                                                            # note 1
run sp-sg      streetview --densify_until_iter 3000 --size_threshold_from_iter 1000
run defree_sfm streetview                                                          # note 1
