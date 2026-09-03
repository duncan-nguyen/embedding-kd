#!/usr/bin/env bash
# EMO. Optimal-transport KD with a learned teacher->student map. Its batch of 4 and lr of 1e-5 are emo_config.py: the transport plan is what makes the batch small.
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/common.sh"

run_distill emo "$@"
