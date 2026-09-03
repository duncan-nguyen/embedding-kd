#!/usr/bin/env bash
# SimCSE-only control. No teacher is loaded -- this is the row every distilled
# row is read against -- so TEACHER_MODEL is cleared before run_distill, which
# then omits the teacher flags and leaves the second visible device unused.
# simcse_view = "dropout" is simcse_config.py.
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/common.sh"

TEACHER_MODEL=""

run_distill simcse "$@"
