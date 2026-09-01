# ICLR Embedding KD

This repository distills compact text embedding students from large embedding
teachers. The reproduction target follows the Qwen3 to BERT-base pair in
`docs/TALAS.pdf`:

```text
Qwen3-Embedding-4B -> BERT-base 109M
```

The default training corpus is 150K unlabeled texts:

```text
data/train_set/train_150k.csv
```

100K of them are a fixed sample of the benchmark train splits (the corpus the
earlier 100K runs used); the other 50K are 25K MS MARCO queries and 25K MS MARCO
passages, drawn from disjoint rows of one pinned shard so no query in the corpus
sits next to a passage that was retrieved for it. No qrel, `is_selected` flag or
answer is ever read -- only raw text, so the objective stays unlabeled. MS MARCO
is a *training* source for the retrieval evaluation and never a test one, which
keeps ArguAna/FiQA/SCIDOCS zero-shot cross-dataset.

### The data-scaling ladder

One script builds every rung. The base sample is identical at every size and the
MS MARCO rows are accepted in a fixed permutation order, so a larger corpus
**extends** a smaller one rather than resampling it -- `train_150k.csv` is a
row-for-row prefix of `train_200k.csv` in all three blocks. The ablation then
varies corpus size alone, not which sentences are in the corpus.

```bash
python3 scripts/download_retrieval_benchmarks.py   # needed for the overlap check
python3 scripts/build_train_corpus.py --total 150000
python3 scripts/build_train_corpus.py --total 200000
```

| Corpus | Base | MS MARCO queries | MS MARCO passages |
| --- | --- | --- | --- |
| `train_150k.csv` | 100,000 | 25,000 | 25,000 |
| `train_200k.csv` | 100,000 | 50,000 | 50,000 |

Everything is seeded and the Hub file is pinned to a commit sha, so a re-run
reproduces each file byte for byte. The matching `train_<n>k.manifest.json`
records the seed, the shard, the per-source counts and every overlap the build
dropped. Every new text is checked, on normalised form, against the 100K base,
against the other new texts, and against the queries and corpora of all three
retrieval benchmarks plus every existing test and validation split.

One MS MARCO shard splits into ~57.7K rows per pool, so a single-shard build tops
out near 215K rows; past that, point `MSMARCO_SHARD` at another of the seven train
shards. For a clean 100K rung, `--total 100000 --out <path>` writes the base
sample alone (the 100,000 rows the other rungs share), which is not the same file
as the 102,361-row `data/train_set/train_100k.csv` it is sampled from.

The TALAS paper setup (about 15K unlabeled sentences from EMOTION, WiC and STS-B)
is still on disk as `data/train_set/merged_3_data_5k_each.csv`.

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
| `geoode` | `config/geoode_config.py` | GeoODE-KD: endpoint distillation onto a frozen PCA+Procrustes teacher map, with an InfoNCE regulariser |
| `cdm` | `config/cdm_config.py` | Contextual dynamic mapping across the two tokenizers |
| `dskd` | `config/dskd_config.py` | Dual-space KD with a learned projection |
| `emo` | `config/emo_config.py` | Optimal-transport embedding distillation |
| `stella` | `config/stella_config.py` | Stella multi-dimension student heads |
| `rkd` | `config/rkd_config.py` | Relational KD: distance-wise and angle-wise relations to the cached teacher |
| `simcse` | `config/simcse_config.py` | SimCSE-only control: the student's contrastive loss, no teacher |

`talas`, `geoode` and `rkd` cache the teacher's sentence embeddings once
(`cache_path`) and free the teacher model afterwards; `simcse` loads no teacher
at all; the remaining methods run the teacher alongside the student every step.

`--teacher_pooling` (`last_token`, `cls`, `mean`) is how the teacher's sentence
vector is read, and it follows the teacher family, not the method: Qwen3-Embedding
is a decoder and reads its last token (the default), BGE-M3 is an XLM-R encoder and
reads `cls`. The cached methods apply it once at cache time and the online ones
apply it every step, so one flag covers all eight methods. The token-level
alignments of `cdm` (and `emo`) strip a sub-word marker before comparing token
strings; `--teacher_special_token` sets it (`Ġ` for Qwen3's byte-level BPE, `▁`
for SentencePiece teachers such as BGE-M3; EMO reads the same flag as the
teacher's BOS token string, `<s>`).

Teacher/student pairs that have been checked against every method:

| Teacher | Student | Flags |
| --- | --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` (1024-d) | `jim12345/MiniLMv2-L6-H384-distilled-from-BERT-Base` (384-d, 6 layers) | `--teacher_pooling last_token --teacher_special_token Ġ` |
| `BAAI/bge-m3` (1024-d, XLM-R) | `nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large` (768-d, 6 layers) | `--teacher_pooling cls --teacher_special_token ▁` |
| `Qwen/Qwen3-Embedding-4B` (2560-d) | `google-bert/bert-base-uncased` (768-d, 12 layers) | `--teacher_pooling last_token --teacher_special_token Ġ` |

Nothing in the objectives is sized to a particular pair: projection heads read
both widths from the model configs, GeoODE-KD's teacher map is fitted to whatever
the two widths are, and TALAS anchors the last `last_layer_idx` of
however many hidden states the student returns. `test_mdd.ipynb` selects one of
the three pairs with its `PAIR` variable and passes the flags above.

The last two rows are the reference points the distillation methods are read
against. `simcse` is the no-distillation control: the same student, corpus,
schedule and pooling as every other row, with the teacher term removed and only
the InfoNCE task loss those objectives already contain left in place, so
whatever a method reports above this line is what its teacher signal bought. Its
positives are two dropout views of the same sentence (unsupervised SimCSE);
`--simcse_view pair` uses the row's paired sentence instead.

`rkd` implements Park et al. (2019): the teacher supervises the *relations*
between examples — pairwise distances normalised by the batch mean, and the
cosines of the angles subtended at each middle point — under a Huber loss. Both
potentials are invariant to the width, scale and orientation of the space they
are measured in, so a 2560-d teacher supervises a 768-d student with nothing
fitted between them and no parameters added, exactly like `geoode`. Its flags are
`--w_dist`/`--w_angle` (the paper's 25 and 50) and `--normalize_student`, which
measures the student's relations on the unit sphere the normalised teacher cache
and every cosine benchmark already live on (`--no-normalize_student` is the
raw-Euclidean ablation). It constrains the final layer only, like `geoode`, which is what makes it the
relational counterpart to `geoode`'s point-wise endpoint term.

`geoode` implements `docs/ode_embedding_kd.pdf`. It reduces the cached teacher
embeddings to the student dimension with a PCA map fitted on the cache itself,
then rotates those coordinates onto the *untrained* student's own embedding space
by orthogonal Procrustes (`P_T = P_PCA R`; both saved in `teacher_projection.pt`
next to the checkpoints). The rotation is a closed-form statistic, not a
parameter: it removes the arbitrary gauge of the PCA basis from the endpoint loss
without touching the Gram matrix (`--no-gauge_align` is the ablation;
`--gauge_refit_every N` re-estimates it against the current student every N
epochs). Both factors of that map are claims, so both have a control. The
subspace: `--projection_type random` draws a Haar-random subspace of the same
rank and `random_gaussian` the Johnson-Lindenstrauss map that gives up
orthonormality too, while `--no-pca_center_fit` is the uncentered-SVD arm in
which the teacher's mean vector may itself be the first retained direction.
Whether the map should be frozen at all: `--projection_type learned_t2s` and
`learned_s2t` replace it with a linear layer trained alongside the student (the
teacher mapped down, or the student mapped up into the teacher's space), which is
what TALAS, LEAF, EMO and sentence-transformers v5.5 do. Those parameters exist
during training only — inference is still the plain student encoder, so what
changes is the supervision rather than the artefact. The
orientation: `--gauge_rotation random` applies a Haar-random rotation of exactly
the Procrustes arm's cost, which is the control that matters, because the PCA
basis is already an arbitrary gauge and so `--no-gauge_align` alone cannot
separate "R is the right orientation" from "R is an orientation". The run prints
and saves what its map actually did — retained energy against what a random
subspace of that rank would retain, the student-target cosine before and after
the rotation, and the participation ratio of the cross-covariance, which says in
advance whether the gauge can matter at all (at PR ~ 1 it can only rotate one
mean vector onto another). `scripts/run_target_map_ablation.py` runs the whole
grid off one shared teacher cache and reads it back as a table. The objective is
`L_end + L_ctr` and nothing else: the final layer is anchored on those targets and
regularised by InfoNCE over two dropout views (`--lambda_end 1`,
`--lambda_ctr 0.5`). Only the endpoint is supervised — no term reads the
intermediate layers, so what the lower stack does with depth is left to the
encoder. Its other flag is `--student_pooling`. Training adds no parameters and
inference is the plain student encoder.

`--lambda_topo` adds the H0 persistence term of `src/criterions/h0_topological_loss.py`
to that objective. It asks the student batch and the teacher batch to have the
same *shape*: the finite death times of the H0 Vietoris-Rips diagram are the
weights of the batch's minimum spanning tree, so matching them sorted is a
statement about how the cloud is connected and about nothing else. Like RKD's
potentials it needs no shared basis and no shared width, which is why it reads
the teacher cache *before* `P_T` narrows it to `d_S` — it is the one term in the
run whose supervision the choice of target map cannot colour, and the reason it
is worth reporting next to `--lambda_gram` (the pairwise-similarity control,
which can only be measured after that map). The MST is Prim's, run on the batch:
the selection is detached and the selected edge weights are not, so the gradient
reaches the endpoints of the tree the batch actually has. `--topo_metric` picks
the ground metric on the sphere (`chord`, the default Euclidean `sqrt(2 - 2cos)`;
`angular`, the geodesic; `cosine`). Death times are O(1) and the term is their
mean squared error, so it enters small — sweep the weight over decades
(`0.01`, `0.1`, `1.0`) rather than around `0.5`. Rung 4 of the structural audit
already measures the same object post-hoc (`h0_w1`, Wasserstein-1 between the two
barcodes on the probe set), which is where to read off how far apart the diagrams
are before spending a run on the weight — and the caveat that comes with it: for
a `--lambda_topo > 0` arm, `h0_w1` is no longer an independent measurement, it is
the training objective scored back. The audit builds its MST under `1 - cos` and
the default term under `chord`; both are increasing in `1 - cos`, so the tree is
the same and only the edge lengths are scaled. `--topo_metric cosine` makes the
two read the same units.

Training is single-process. Two visible CUDA devices place the student on
`cuda:0` and the teacher on `cuda:1`; one device puts both on `cuda:0`.

## Run

From the repo root:

```bash
source .venv/bin/activate
bash scripts/train_talas.sh
```

One script per method lives in `scripts/` (`train_talas.sh`, `train_cdm.sh`,
`train_dskd.sh`, `train_emo.sh`, `train_stella.sh`, `train_geoode.sh`,
`train_rkd.sh`, `train_simcse.sh`, plus `.ps1` equivalents).

Or run the Python entry point directly:

```bash
python3 main.py \
  --method talas \
  --train_data data/train_set/train_150k.csv \
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
```

Validation is evaluated and printed after every epoch. Test is evaluated and
printed once after training. The Colab notebook exports the two splits
separately:

```text
models/talas/qwen3_4b_to_bert_base/eval_by_epoch.csv
models/talas/qwen3_4b_to_bert_base/final_test_results.csv
```

## Reusing the Teacher Cache

Encoding the corpus with the teacher is the most expensive step in the pipeline
and it does not change between runs of the same pair, so it should be paid for
once. Pass `--cache_dir` and the run derives the filename from everything a
cache's reusability depends on — teacher, pooling, normalisation, `max_length`
and the corpus *contents*:

```bash
python3 main.py --method geoode --cache_dir runs/teacher_cache ...
# runs/teacher_cache/qwen-qwen3-embedding-4b__train_100k__last_token__8f3b4afdb247.pt
```

One directory then holds every cache the project builds, and a run either finds
exactly its own or misses. Point it somewhere that outlives a single run (on
Colab, a directory in the mounted Drive) and every later run of the same pair —
the other cached methods, the ablation grid, a rerun tomorrow — skips the teacher
entirely. `--cache_path` still names one file directly when you want that.

The pass that builds a cache is forward-only, so it is sized apart from training:
`--cache_batch_size` (default 128) is what the teacher encodes at a time, and
`--batch_size` stays the student's. Batches are formed over length-sorted rows
rather than in corpus order, which on `train_150k` is 2.1x (batch 32) to 2.5x
(batch 128) fewer padded tokens for the teacher to attend over; the cache is still
written back in corpus order, so row *i* is the embedding of row *i*. Drop
`--cache_batch_size` to 0 to fall back to the training batch size on a small card.

## Rebuilding Caches

Each file records the teacher name, pooling, normalisation, `max_length` and a
digest of the corpus it was built from next to the embeddings, and a run whose
settings differ from that record (or whose teacher width or corpus length differs
from the tensor) is rejected at startup with the mismatching fields listed,
rather than trained against the wrong teacher. This is what catches a swap
between two teachers of the same width, e.g. Qwen3-Embedding-0.6B and BGE-M3,
both 1024-d — and, via the digest, a corpus regenerated to the same path with the
same row count, which nothing else can see. With `--cache_dir` those differences
change the filename instead, so they are a miss rather than a refusal. To rebuild
a cache, remove the file and rerun:

```bash
rm cache/talas/qwen3_4b_bert_base_teacher_train.pt
```

Caches written before this record existed still load; they are checked by shape
only and a warning says so.

## Evaluation

All three families score frozen student embeddings (CLS pooling), and each handles
calibration differently:

| Family | Benchmarks | How a score is produced |
| --- | --- | --- |
| Classification | Banking77, Emotion, Tweet | a logistic-regression probe is fitted on that benchmark's *train* split and scored on the eval split (accuracy, macro-F1) |
| Pair | MRPC, SciTail, WiC | cosine similarity mapped to `[0, 1]`, then a **decision threshold** turns it into a label (accuracy, F1, precision, recall, average precision) |
| STS | SICK, STS12, STS-B | cosine similarity against gold scores, Spearman correlation |
| Retrieval | ArguAna, FiQA-2018, SCIDOCS | exhaustive cosine ranking of the whole corpus (nDCG@10, Recall@10, MRR@10) |

The summary block reports `AVG (IOD)` and `AVG (OOD)` over the sentence-level
families only, `AVG (RETRIEVAL)` over the three retrieval benchmarks, and
`AVG (ALL)` over all twelve, so adding retrieval did not redefine the two
averages earlier runs are reported against.

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

Zero-shot retrieval:

```text
ArguAna, FiQA-2018, SCIDOCS
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

### Retrieval benchmarks

The three retrieval benchmarks are ~90 MB and are not tracked in git. Fetch them
once, before the first run:

```bash
python3 scripts/download_retrieval_benchmarks.py
```

They land under `data/test_set/retrieval/<name>/{corpus,queries,qrels}.csv`, pinned
to a Hub commit sha, with row counts asserted against the BEIR paper (ArguAna
8674/1406, FiQA 57638/648, SCIDOCS 25657/1000).

Scoring follows BEIR exactly, so the numbers are comparable to published ones: a
document is `title + " " + text`, ranking is exhaustive cosine over the full
corpus with no ANN index, a document whose id equals the query id is dropped
before the top-k cut, and nDCG@10 uses `2^rel - 1` gains with `log2(rank + 1)`
discounts. Checked end to end against MTEB: `all-MiniLM-L6-v2` (mean-pooled)
scores **50.25** on ArguAna here against MTEB's 50.17.

Retrieval is scored on the test split only -- there are no validation qrels for
these three, and embedding the three corpora is ~92k forward passes, more than the
rest of the protocol combined. To skip it:

```bash
python3 main.py --method geoode --no_eval_retrieval
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
