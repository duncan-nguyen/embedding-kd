#!/usr/bin/env bash
# RKD. Relational KD: the distance and angle potentials of the batch, so it needs
# no shared basis and no shared width. w_dist = 25, w_angle = 50 and lr = 7e-5
# are rkd_config.py.
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/common.sh"

run_distill rkd \
    --cache_dir "$CACHE_DIR" \
    "$@"
