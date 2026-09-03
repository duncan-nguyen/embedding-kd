# RKD. Relational KD: the distance and angle potentials of the batch, so it
# needs no shared basis and no shared width. w_dist = 25, w_angle = 50 and
# lr = 7e-5 are rkd_config.py.
#
# Extra flags for main.py are forwarded from this script's own arguments, so a
# one-off override wins on a repeat:
#
#     .\scripts\methods\rkd\train.ps1 --epochs 3 --no_wandb

. (Join-Path $PSScriptRoot "..\..\lib\common.ps1")

Invoke-Distill -Method rkd -Extra (@(
    "--cache_dir", $CACHE_DIR
) + $args)
