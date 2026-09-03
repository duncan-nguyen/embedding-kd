# Stella. Two-stage: epochs_stage1 + epochs_stage2 in stella_config.py,
# not one --epochs.
#
# Extra flags for main.py are forwarded from this script's own arguments, so a
# one-off override wins on a repeat:
#
#     .\scripts\methods\stella\train.ps1 --epochs 3 --no_wandb

. (Join-Path $PSScriptRoot "..\..\lib\common.ps1")

Invoke-Distill -Method stella -Extra $args
