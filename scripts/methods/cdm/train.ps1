# CDM. Token-level cross-model distillation over a DTW alignment;
# w_task and alpha_dtw are cdm_config.py.
#
# Extra flags for main.py are forwarded from this script's own arguments, so a
# one-off override wins on a repeat:
#
#     .\scripts\methods\cdm\train.ps1 --epochs 3 --no_wandb

. (Join-Path $PSScriptRoot "..\..\lib\common.ps1")

Invoke-Distill -Method cdm -Extra $args
