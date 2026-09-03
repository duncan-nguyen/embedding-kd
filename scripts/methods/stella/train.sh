#!/usr/bin/env bash
# Stella. Two-stage: epochs_stage1 + epochs_stage2 in stella_config.py, not one --epochs.
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/common.sh"

run_distill stella "$@"
