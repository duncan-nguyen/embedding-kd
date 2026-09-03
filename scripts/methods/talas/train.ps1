# TALAS. The main baseline: a learned student->teacher map trained against the
# cached teacher vectors. Its w_task = 0.001 is talas_config.py.
#
# Extra flags for main.py are forwarded from this script's own arguments, so a
# one-off override wins on a repeat:
#
#     .\scripts\methods\talas\train.ps1 --epochs 3 --no_wandb

. (Join-Path $PSScriptRoot "..\..\lib\common.ps1")

Invoke-Distill -Method talas -Extra (@(
    "--cache_dir", $CACHE_DIR
) + $args)
