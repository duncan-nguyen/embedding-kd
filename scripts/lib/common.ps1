# Shared setup for every scripts\methods\<method>\train.ps1. Dot-sourced, never run.
#
# The PowerShell twin of common.sh, and deliberately the same shape: it resolves
# the repo root from its own location and runs there, so a method script works
# from any working directory.
#
# What lives here is only what every method shares: the pair, the corpus, the
# teacher cache and the two environment variables the run reads. Everything a
# method alone decides -- its loss weights, its batch size, its epochs -- is left
# to config\<method>_config.py. Every CLI flag of main.py defaults to None and a
# None never overrides the config, so a value omitted below is the config's, not
# a silent default.
#
# Every value is overridable from the environment, no file edited:
#
#     $env:TRAIN_DATA = "data/train_set/train_200k.csv"
#     .\scripts\methods\talas\train.ps1
#
# and any argument given to a method script is forwarded to main.py after the
# flags the script itself sets, so it wins on a repeat:
#
#     .\scripts\methods\geoode\train.ps1 --lambda_topo 0.1 --no_wandb

$ErrorActionPreference = "Stop"

$REPO_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Get-Default([string] $Name, [string] $Fallback) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrEmpty($value)) { return $Fallback }
    return $value
}

# Training is single-process. Two visible CUDA devices place the student on
# cuda:0 and the teacher on cuda:1; one device puts both on cuda:0.
$env:CUDA_VISIBLE_DEVICES = Get-Default "CUDA_VISIBLE_DEVICES" "0,1"
$env:TOKENIZERS_PARALLELISM = Get-Default "TOKENIZERS_PARALLELISM" "false"

# The pair the main tables are built on. TEACHER_POOLING is the teacher's
# sentence vector: Qwen3-Embedding is a decoder, so it is the last token; an
# encoder teacher such as BGE-M3 reads "cls". The cached methods apply it once,
# at cache time, so a wrong value here is baked into the cache rather than
# caught at the first step.
$STUDENT_MODEL = Get-Default "STUDENT_MODEL" "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
$TEACHER_MODEL = Get-Default "TEACHER_MODEL" "Qwen/Qwen3-Embedding-0.6B"
$TEACHER_POOLING = Get-Default "TEACHER_POOLING" "last_token"

# Sub-word marker the token-level alignments (cdm/emo) strip before comparing
# token strings: the byte-level BPE marker for Qwen3, the SentencePiece one for
# a teacher such as BGE-M3 (the literal characters are in common.sh, which is
# UTF-8; keep this file ASCII so Windows PowerShell reads it the same either
# way). Empty leaves it to the config, which is the Qwen3
# value -- set it when switching teacher family, or cdm and emo align against a
# marker their teacher never emits.
$TEACHER_SPECIAL_TOKEN = Get-Default "TEACHER_SPECIAL_TOKEN" ""

$TRAIN_DATA = Get-Default "TRAIN_DATA" "data/train_set/train_150k.csv"

# Re-encoding the corpus with the teacher is the most expensive thing in the
# pipeline and never changes between runs of the same pair, so the cache lives
# outside the run. Read by the cached methods (talas/geoode/rkd) only.
$CACHE_DIR = Get-Default "CACHE_DIR" "cache/teacher"

# Under checkpoints\, which is gitignored.
$CHECKPOINT_ROOT = Get-Default "CHECKPOINT_ROOT" "checkpoints"

# Invoke-Distill -Method <method> -Extra <array of main.py flags>
#
# -Extra is one array rather than loose arguments: every flag main.py takes
# starts with "--", which PowerShell would otherwise try to bind as a parameter
# of this function.
#
# The teacher flags are omitted when $TEACHER_MODEL is empty, which is how the
# simcse control -- the one run that loads no teacher -- reuses this function.
function Invoke-Distill {
    param(
        [Parameter(Mandatory = $true)] [string] $Method,
        [string[]] $Extra = @()
    )

    $teacherLabel = if ([string]::IsNullOrEmpty($TEACHER_MODEL)) { "<none>" } else { $TEACHER_MODEL }
    Write-Host "======================================"
    Write-Host "Training with $Method method"
    Write-Host "  student: $STUDENT_MODEL"
    Write-Host "  teacher: $teacherLabel"
    Write-Host "  corpus:  $TRAIN_DATA"
    Write-Host "======================================"

    $cliArgs = @(
        "--method", $Method,
        "--train_data", $TRAIN_DATA,
        "--student_model", $STUDENT_MODEL,
        "--save_dir", "$CHECKPOINT_ROOT/$Method"
    )
    if (-not [string]::IsNullOrEmpty($TEACHER_MODEL)) {
        $cliArgs += @("--teacher_model", $TEACHER_MODEL, "--teacher_pooling", $TEACHER_POOLING)
    }
    if (-not [string]::IsNullOrEmpty($TEACHER_SPECIAL_TOKEN)) {
        $cliArgs += @("--teacher_special_token", $TEACHER_SPECIAL_TOKEN)
    }
    if ($Extra) { $cliArgs += $Extra }

    Push-Location $REPO_ROOT
    try {
        python main.py @cliArgs
        if ($LASTEXITCODE -ne 0) { throw "main.py exited with $LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
}
