# EMO. Optimal-transport KD with a learned teacher->student map. Its batch
# of 4 and its lr of 1e-5 are emo_config.py: the transport plan is what
# makes the batch small.
#
# Extra flags for main.py are forwarded from this script's own arguments, so a
# one-off override wins on a repeat:
#
#     .\scripts\methods\emo\train.ps1 --epochs 3 --no_wandb

. (Join-Path $PSScriptRoot "..\..\lib\common.ps1")

Invoke-Distill -Method emo -Extra $args
