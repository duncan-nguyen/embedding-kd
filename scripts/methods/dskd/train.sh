#!/usr/bin/env bash
# DSKD. Dual-space KD. Its 10 epochs and its w_task = alpha_dtw = 1.0 are dskd_config.py.
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/common.sh"

run_distill dskd "$@"
