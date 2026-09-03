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
