#!/usr/bin/env bash
# CDM. Token-level cross-model distillation over a DTW alignment; w_task and alpha_dtw are cdm_config.py.
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/common.sh"

run_distill cdm "$@"
