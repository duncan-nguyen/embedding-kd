# SimCSE-only control. No teacher is loaded -- this is the row every distilled
# row is read against -- so $TEACHER_MODEL is cleared before Invoke-Distill,
# which then omits the teacher flags and leaves the second visible device
# unused. simcse_view = "dropout" is simcse_config.py.
#
# Extra flags for main.py are forwarded from this script's own arguments, so a
# one-off override wins on a repeat:
#
#     .\scripts\methods\simcse\train.ps1 --epochs 3 --no_wandb

. (Join-Path $PSScriptRoot "..\..\lib\common.ps1")

$TEACHER_MODEL = ""

Invoke-Distill -Method simcse -Extra $args
