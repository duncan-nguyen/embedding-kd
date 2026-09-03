#!/usr/bin/env bash
# TALAS. The main baseline: a learned student->teacher map trained against the
# cached teacher vectors. Its w_task = 0.001 is talas_config.py.
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/common.sh"

run_distill talas \
    --cache_dir "$CACHE_DIR" \
    "$@"
