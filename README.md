# ICLR Embedding KD

This repository distills compact text embedding students from large embedding
teachers. The reproduction target follows the Qwen3 to BERT-base pair in
`docs/TALAS.pdf`:

```text
Qwen3-Embedding-4B -> BERT-base 109M
```

The training corpus follows the TALAS paper setup: about 15K unlabeled sentences
sampled from the three in-domain datasets EMOTION, WiC, and STS-B. In this repo,
that corpus is:

```text
data/train_set/merged_3_data_5k_each.csv
```

Benchmark CSV files are separated under `data/train_set/`, `data/val_set/`,
and `data/test_set/`. Classification probe train and validation files are
checked for normalized-text leakage before evaluation.

## Environment

Do not install packages into the global Python environment. Create and use a
project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Weights & Biases logging:

```bash
wandb login
```

If you want a local/offline W&B run:

```bash
export WANDB_MODE=offline
```

## Methods

`--method` selects the distillation objective. Each one has its own config class
under `config/`, which supplies the defaults that the CLI flags then override:

| `--method` | Config | Objective |
| --- | --- | --- |
| `talas` | `config/talas_config.py` | Teacher-anchor KD on cached teacher embeddings, with a structural term and SAM |
| `geoode` | `config/geoode_config.py` | GeoODE-KD: student layers trained as Euler steps of a teacher-guided flow on the hypersphere |
| `cdm` | `config/cdm_config.py` | Contextual dynamic mapping across the two tokenizers |
| `dskd` | `config/dskd_config.py` | Dual-space KD with a learned projection |
| `emo` | `config/emo_config.py` | Optimal-transport embedding distillation |
| `stella` | `config/stella_config.py` | Stella multi-dimension student heads |

`talas` and `geoode` cache the teacher's sentence embeddings once (`cache_path`)
and free the teacher model afterwards; the other methods run the teacher
alongside the student every step.

`geoode` implements `docs/ode_embedding_kd.pdf`. It reduces the cached teacher
embeddings to the student dimension with a PCA map fitted on the cache itself,
then rotates those coordinates onto the *untrained* student's own embedding space
by orthogonal Procrustes (`P_T = P_PCA R`; both saved in `teacher_projection.pt`
next to the checkpoints). The rotation is a closed-form statistic, not a
parameter: it removes the arbitrary gauge of the PCA basis from the endpoint loss
without touching the Gram matrix (`--no-gauge_align` is the ablation). It then
supervises each
Transformer layer as one Riemannian Euler step of a teacher-conditioned flow
instead of pushing every layer at the final teacher embedding. Its own flags are
`--alpha`/`--beta` (the semantic and relational parts of the energy),
`--lambda_end`/`--lambda_dyn`/`--lambda_ctr` (the three loss weights),
`--guidance_schedule`/`--guidance_power` (the depth schedule s(t)) and
`--student_pooling`. Training adds no parameters and inference is the plain
student encoder.

The flow is the finite-horizon form of the paper's energy: the semantic term is
the squared geodesic distance to the teacher, so its negative Riemannian gradient
is the sphere's log map, and the field is run in the time warp `s(t) / R(t)` with
`R(t) = int_t^1 s`. Under that warp the instance-only flow contracts the geodesic
distance as `d(t) = d(0) R(t) / R(0)` and reaches the teacher exactly at `t = 1`
(a plain gradient flow only gets there as `t -> inf`, which is why an earlier
draft saw teacher cosine stay flat through depth and jump in the last two layers).
The per-layer target is therefore `Exp_Z(rho_l V)` with
`rho_l = 1 - R(t_{l+1}) / R(t_l)`, which with `beta = 0` is the spherical
interpolation slerp(Z, T; rho_l); the last `rho` is 1, so `L_end` is the boundary
condition of the same flow rather than a competing loss. The field is taken from
the per-sample energy (`B` times the batch mean the paper writes) so its speed
does not depend on batch size. `depth_metrics.jsonl` records `field_norm` next to
`step_norm` and the closed-form `predicted_geodesic_distance` next to the
realized one, so how closely the student tracks the flow stays visible.

Training is single-process. Two visible CUDA devices place the student on
`cuda:0` and the teacher on `cuda:1`; one device puts both on `cuda:0`.

## Run

From the repo root:

```bash
source .venv/bin/activate
bash scripts/train_talas.sh
```

One script per method lives in `scripts/` (`train_talas.sh`, `train_cdm.sh`,
`train_dskd.sh`, `train_emo.sh`, `train_stella.sh`, `train_geoode.sh`, plus
`.ps1` equivalents).

Or run the Python entry point directly:

```bash
python3 main.py \
  --method talas \
  --train_data data/train_set/merged_3_data_5k_each.csv \
  --student_model google-bert/bert-base-uncased \
  --teacher_model Qwen/Qwen3-Embedding-4B \
  --batch_size 32 \
  --epochs 5 \
  --lr 2e-5 \
  --max_length 256 \
  --save_dir models/talas/qwen3_4b_to_bert_base
```

To disable W&B:

```bash
python3 main.py --method talas --no_wandb
```

To persist student weights after every epoch, provide a durable directory:

```bash
python3 main.py --method talas \
  --weights_dir "/content/drive/MyDrive/[ICLR] Embedding KD/weights/qwen3_4b_to_bert_base"
```

`test_mdd.ipynb` is the Colab/local Jupyter runner for all six methods using
`Qwen/Qwen3-Embedding-4B -> google-bert/bert-base-uncased`. It runs each method
in a separate process and writes per-method logs, checkpoints, comparison tables
and figures to Google Drive (or `runs/` outside Colab).

## Outputs

Model checkpoints and weights are saved under `--save_dir`, e.g.:

```text
models/talas/qwen3_4b_to_bert_base/
```

Training and benchmark metrics are written to:

```text
models/talas/qwen3_4b_to_bert_base/metrics.jsonl     # one record per epoch
models/talas/qwen3_4b_to_bert_base/step_metrics.jsonl # one record per optimizer step
models/talas/qwen3_4b_to_bert_base/depth_metrics.jsonl # per-layer profile, sampled
```

## Depth Diagnostics

`talas` and `geoode` additionally sample a per-layer profile every
`--depth_log_every` steps (default 50, `0` disables) and append it to
`depth_metrics.jsonl`. A compact table is printed at the end of every epoch. Both
methods are measured with the same parameter-free probe, so their profiles are
directly comparable — which is the point, since GeoODE-KD's central claim is
about how the depth profile differs from static multi-layer anchoring.

Each record holds, for one batch: teacher cosine, relational (Gram) gap and
energy at every depth; the ODE consistency residual, the prescribed step size
`|dt*F|`, the realized step size `|dz|` and their direction alignment at every
transition; plus counts of depths where a curve moves the wrong way. The last
group matters most for reading a run: the residual alone is small whenever the
layers barely move, and only `|dz|` next to `|dt*F|` and the alignment separate
"follows the teacher's direction" from "ignores a negligible field".

After training, turn the JSONL into figures and a summary table:

```bash
python3 scripts/plot_depth_diagnostics.py runs/<stamp>/geoode runs/<stamp>/talas
```

Passing several runs (or one parent directory) overlays their final epochs into
`comparison_depth_*.png`; each run also gets its own per-epoch curves,
`*_depth_progress.png` over training steps, `*_loss_components.png`, plus
`depth_summary.csv` and `depth_curves.csv`. Cell 8 of `test_mdd.ipynb` runs this
step and displays the figures inline.

Validation is evaluated and printed after every epoch. Test is evaluated and
printed once after training. The Colab notebook exports the two splits
separately:

```text
models/talas/qwen3_4b_to_bert_base/eval_by_epoch.csv
models/talas/qwen3_4b_to_bert_base/final_test_results.csv
```

The teacher embedding cache is written to `--cache_path`, e.g.:

```text
cache/talas/qwen3_4b_bert_base_teacher_train.pt
```

## Rebuilding Caches

The cache is keyed only by its path, so a change of training corpus, teacher
model or pooling needs the old file removed first:

```bash
rm cache/talas/qwen3_4b_bert_base_teacher_train.pt
```

Then rerun training. A cache whose row count does not match the corpus is
rejected at startup rather than silently misaligned.

## Evaluation

All three families score frozen student embeddings (CLS pooling), and each handles
calibration differently:

| Family | Benchmarks | How a score is produced |
| --- | --- | --- |
| Classification | Banking77, Emotion, Tweet | a logistic-regression probe is fitted on that benchmark's *train* split and scored on the eval split (accuracy, macro-F1) |
| Pair | MRPC, SciTail, WiC | cosine similarity mapped to `[0, 1]`, then a **decision threshold** turns it into a label (accuracy, F1, precision, recall, average precision) |
| STS | SICK, STS12, STS-B | cosine similarity against gold scores, Spearman correlation |

Validation runs after every `--eval_every` epochs; test runs once, after training.
`--evaluate_test_each_epoch` swaps that around: the test split is evaluated after
every `--eval_every` epochs and no validation pass runs at all. It implies
`--pair_threshold_source test` (a run with no validation has nowhere else to get a
threshold from), and the two are checked for consistency before the models load
rather than at the end of the first epoch. The end-of-run table then reuses the
last epoch's evaluation instead of repeating it.

With that flag, every reported number has been seen during model selection: the
epoch you pick and the threshold you sweep both read the test labels. Keep it off
for any number you intend to publish.

The pair threshold is the only quantity carried between splits. By default it is
swept over 200 candidates on the validation split and reused unchanged on test, so
the test score stays held out; a test evaluation with no preceding validation is
refused rather than silently calibrated. To sweep it on the test split instead:

```bash
python3 main.py --method geoode --pair_threshold_source test
```

Then the pair accuracy/F1/precision/recall become an **upper bound** rather than a
held-out estimate, since the threshold is chosen on the labels being scored.
`average_precision` (the primary pair metric in the summary table) is
threshold-free and unaffected either way, as are the classification and STS
families. The classification probe is always fitted on that benchmark's own
`train` split, never on validation or test. Runs using it are marked in the printed table and record
`"pair_threshold_source": "test"` in `metrics.jsonl`. It also removes test's
dependency on a validation pass, so it can be combined with a large
`--eval_every` to skip per-epoch validation entirely.

## Benchmarks

The training loop evaluates these benchmark groups:

Classification:

```text
Banking77, Emotion, Tweet
```

Pair classification:

```text
MRPC, SciTail, WiC
```

Semantic textual similarity:

```text
SICK, STS12, STS-B
```

Validation runs after each epoch. Final test evaluation runs after training,
reusing pair-classification thresholds selected on validation.

Only the three classification benchmarks read a train split: their score comes
from a logistic-regression probe fitted on train embeddings. The pair and STS
benchmarks are scored without fitting anything, so their train splits are on
disk for completeness only. To (re)fetch them:

```bash
python3 scripts/download_eval_train_splits.py            # skips what exists
python3 scripts/download_eval_train_splits.py --force    # refetch everything
```

The script rebuilds each benchmark's existing validation and test files from the
upstream source first and refuses to write the train split unless they match, so
a source that has drifted fails loudly instead of landing a mismatched split.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
