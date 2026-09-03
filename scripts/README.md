# scripts/

```
lib/          common.sh, common.ps1 -- everything the eight training runs share
methods/      one folder per distillation method: train.sh + train.ps1
data/         builds the training corpus and downloads the benchmarks
ablations/    the target-map grid
figures/      paper-figure mockups (synthetic data, never evidence)
```

## Training

From anywhere -- each script resolves the repo root from its own location:

```bash
bash scripts/methods/talas/train.sh
bash scripts/methods/geoode/train.sh
```

```powershell
.\scripts\methods\talas\train.ps1
```

One folder per method under `methods/`: `cdm`, `dskd`, `emo`, `geoode`, `rkd`,
`simcse`, `stella`, `talas`.

### What a method script contains, and what it does not

A method script holds only what that method alone decides — its cache, its
instrumentation — and nothing else. It does **not** restate the method's batch
size, epochs, learning rate or loss weights: those live in
`config/<method>_config.py`, every CLI flag of `main.py` defaults to `None`, and
a `None` never overrides the config. So a value absent from a script is the
config's value, not a silent default, and there is one place to change it.

That is why the eight scripts differ so little. `dskd` trains for 10 epochs,
`emo` at batch 4 and lr 1e-5, `stella` in two stages with no single `--epochs`,
`rkd` at lr 7e-5 — none of that is written here, and all of it still happens.

### Overriding

Environment variables, read by both `lib/common.sh` and `lib/common.ps1`:

| variable | default | note |
| --- | --- | --- |
| `STUDENT_MODEL` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base` | |
| `TEACHER_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | |
| `TEACHER_POOLING` | `last_token` | `cls` for an encoder teacher such as BGE-M3 |
| `TEACHER_SPECIAL_TOKEN` | unset | set when switching teacher family; read by `cdm`/`emo` |
| `TRAIN_DATA` | `data/train_set/train_150k.csv` | |
| `CACHE_DIR` | `cache/teacher` | read by `talas`/`geoode`/`rkd` |
| `CHECKPOINT_ROOT` | `checkpoints` | |
| `CUDA_VISIBLE_DEVICES` | `0,1` | student on `cuda:0`, teacher on `cuda:1`; one device puts both on `cuda:0` |

```bash
TRAIN_DATA=data/train_set/train_200k.csv bash scripts/methods/talas/train.sh
TEACHER_MODEL=BAAI/bge-m3 TEACHER_POOLING=cls \
    STUDENT_MODEL=nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large \
    bash scripts/methods/geoode/train.sh
```

Anything else goes straight through as a `main.py` flag, appended after the ones
the script sets, so it wins on a repeat:

```bash
bash scripts/methods/geoode/train.sh --lambda_topo 0.1 --epochs 3 --no_wandb
```

## The main-results sweep

`experiments/run_main_results.py` is `notebooks/00_main_results.ipynb` as a
headless script, extended from one pair to all three. Same epochs, batch sizes,
learning rates and per-pair sub-word markers — `tests/test_main_results_sweep.py`
compares all 63 commands against the notebook's own builder, so the two cannot
drift apart silently.

```bash
bash scripts/experiments/run_main_results.sh --dry-run   # the plan, run nothing
bash scripts/experiments/run_main_results.sh             # 3 pairs x 7 methods x 3 seeds
```

The `.sh` is the front end: it picks the repo's own interpreter, names the run up
front so the whole sweep is teed into `sweep.log` next to its results, and prints
the resume command if it stops early. Every argument goes through to the Python,
which is where the settings and the resume logic live — call
`python3 scripts/experiments/run_main_results.py` directly when that is what you
want.

7 methods × 3 seeds × 3 pairs = **63 runs**, so it is meant to be started once
and left alone. A finished run is a resume boundary: re-running with the same
`--run-name` skips whatever already has a final-test record, which makes a
crash, an OOM or a Ctrl-C cheap to recover from.

```bash
RUN_NAME=<earlier run> bash scripts/experiments/run_main_results.sh              # resume
bash scripts/experiments/run_main_results.sh --pairs qwen3_0.6b_to_minilm_h384   # one pair
bash scripts/experiments/run_main_results.sh --methods geoode talas --seeds 42 43
python3 scripts/experiments/run_main_results.py --run-name <run> --aggregate-only  # tables only
```

It runs for hours in the foreground, so start it detached:

```bash
tmux new -s sweep 'bash scripts/experiments/run_main_results.sh'
nohup bash scripts/experiments/run_main_results.sh > /dev/null 2>&1 &   # sweep.log has everything
```

`--keep-going` carries on past a failed job instead of stopping; a job that died
mid-run is not restarted by default, because appending to its `metrics.jsonl`
would interleave two runs — `--retry-unfinished` moves the stale directory aside
(it is never deleted) and runs it again.

### What it saves

Under `runs/<run name>/`:

| file | contents |
| --- | --- |
| `run_status.csv` | rewritten after every job, so a killed sweep still leaves a record |
| `<pair>/<method>/seed_<seed>/` | what `main.py` writes, plus `train.log` and `runner_timing.json` |
| `<pair>/final_test_by_seed.csv` | one row per (method, seed), raw `[0, 1]` |
| `<pair>/final_test_mean_std{,_paper}.csv`, `.tex` | mean ± sample std over the seeds, paper scale |
| `<pair>/timing_by_seed.csv` | training time per method per seed |
| `<pair>/efficiency_{by_seed,mean_std}.csv`, `table_3_efficiency.{csv,tex}` | the efficiency table |
| `all_pairs_{final_test_by_seed,mean_std,timing_by_seed}.csv` | the three pairs stacked |

Timing is two clocks, because they answer different questions.
`train_gpu_minutes` is the sum of `step_seconds` — the optimisation itself, and
the one to compare across methods. `wall_minutes` is what the sweep cost end to
end, including tokenisation, the final evaluation and, on the first cached
method of a pair, building the teacher cache. It is written to
`runner_timing.json` as each run finishes, so it survives a resume in a later
process.

A pair is aggregated only once every one of its runs has a final-test record: a
mean ± std over two of three seeds is a different table, so the sweep reports the
gap instead of publishing it. Timings are still written for an unfinished pair.

## Everything else

```bash
python3 scripts/data/download_retrieval_benchmarks.py     # ArguAna, FiQA, SCIDOCS
python3 scripts/data/download_eval_train_splits.py        # train splits of the eval sets
python3 scripts/data/build_train_corpus.py --total 150000 # the distillation corpus
python3 scripts/data/build_merged_all.py                  # every train split, uncapped

python3 scripts/ablations/run_target_map_ablation.py --pair qwen3_4b_to_bert_base
python3 scripts/figures/render_mock_paper_figures.py
```

Each of those carries its own usage in its docstring. Run them from the repo
root or anywhere else — like the training scripts, they locate the repo from
`__file__`.
