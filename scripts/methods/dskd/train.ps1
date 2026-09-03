# DSKD. Dual-space KD. Its 10 epochs and its w_task = alpha_dtw = 1.0
# are dskd_config.py.
#
# Extra flags for main.py are forwarded from this script's own arguments, so a
# one-off override wins on a repeat:
#
#     .\scripts\methods\dskd\train.ps1 --epochs 3 --no_wandb

. (Join-Path $PSScriptRoot "..\..\lib\common.ps1")

Invoke-Distill -Method dskd -Extra $args
