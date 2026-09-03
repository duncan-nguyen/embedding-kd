# GeoODE-KD. The recipe: L_end + L_ctr against frozen spectral targets
# (lambda_end = 1, lambda_ctr = 0.5 in geoode_config.py) and nothing else.
#
# Only the instrumentation is set here, because it is the one thing a bare run
# would leave off. Neither flag changes what is optimised -- a seeded run
# follows the same trajectory with both on -- so both are safe to leave on
# inside an ablation.
#   --diag_every   per-term gradient norms, batch effective ranks, the signed
#                  H0 death-time residual and the student's own H1 diagram.
#   --probe_every  the structural ladder on a fixed probe of corpus sentences,
#                  into probe_metrics.jsonl. 0 turns it off.
#
# Extra flags for main.py are forwarded from this script's own arguments, so a
# one-off override wins on a repeat:
#
#     .\scripts\methods\geoode\train.ps1 --epochs 3 --no_wandb

. (Join-Path $PSScriptRoot "..\..\lib\common.ps1")

$DIAG_EVERY = Get-Default "DIAG_EVERY" "50"
$PROBE_EVERY = Get-Default "PROBE_EVERY" "200"
$PROBE_SIZE = Get-Default "PROBE_SIZE" "1024"

Invoke-Distill -Method geoode -Extra (@(
    "--cache_dir", $CACHE_DIR,
    "--diag_every", $DIAG_EVERY,
    "--probe_every", $PROBE_EVERY,
    "--probe_size", $PROBE_SIZE
) + $args)
