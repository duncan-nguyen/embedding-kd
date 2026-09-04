"""The main-results sweep must stay the notebook it was lifted from.

``scripts/experiments/run_main_results.py`` runs the same jobs as
``notebooks/00_main_results.ipynb`` over all three pairs instead of one. That is
only true while its settings block and its command builder agree with the
notebook's, and nothing but a test notices when one of the two is edited alone --
the numbers would still come out, they would just no longer be comparable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "00_main_results.ipynb"
SWEEP = REPO_ROOT / "scripts" / "experiments" / "run_main_results.py"


def _load_sweep():
    spec = importlib.util.spec_from_file_location("run_main_results", SWEEP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook_cells():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return {cell["id"]: "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"}


@pytest.fixture(scope="module")
def sweep():
    return _load_sweep()


@pytest.fixture(scope="module")
def notebook_settings():
    """Cell 1 of the notebook, with only the wall-clock stamp pinned."""
    source = _notebook_cells()["configuration"].replace(
        'RUN_STAMP = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d-%H%M%S")',
        'RUN_STAMP = "stamp"',
    )
    namespace: dict = {}
    exec(compile(source, "<notebook-configuration>", "exec"), namespace)
    return namespace


def test_the_sweep_settings_are_the_notebook_settings(sweep, notebook_settings):
    assert sweep.PAIRS == notebook_settings["PAIRS"]
    assert sweep.METHOD_SETTINGS == notebook_settings["METHOD_SETTINGS"]
    assert sweep.DATASETS == notebook_settings["DATASETS"]
    # SEEDS is the one setting that is a per-run choice rather than the recipe: the
    # notebook edits it in place (one seed for a smoke run, three for the paper table)
    # while the script takes it as --seeds and keeps its own default grid. What has to
    # hold is that the script can be asked for exactly the seeds the notebook ran.
    notebook_seeds = notebook_settings["SEEDS"]
    assert sweep.parse_args(
        ["--seeds", *(str(seed) for seed in notebook_seeds)]
    ).seeds == notebook_seeds
    assert sweep.DEFAULT_DATASET == notebook_settings["DATASET"]
    assert sweep.EPOCHS == notebook_settings["EPOCHS"]
    assert sweep.MAX_LENGTH == notebook_settings["MAX_LENGTH"]
    assert sweep.NUM_WORKERS == notebook_settings["NUM_WORKERS"]
    assert sweep.EVAL_EVERY == notebook_settings["EVAL_EVERY"]
    assert sweep.EVAL_RETRIEVAL == notebook_settings["EVAL_RETRIEVAL"]
    assert sweep.HOLD_OUT_VALIDATION == notebook_settings["HOLD_OUT_VALIDATION"]
    assert sweep.CUDA_VISIBLE_DEVICES == notebook_settings["CUDA_VISIBLE_DEVICES"]
    assert sweep.GEOODE_H0_BATCH_SIZE == notebook_settings["GEOODE_H0_BATCH_SIZE"]


def test_the_sweep_covers_every_baseline_and_the_paper_method(sweep):
    assert set(sweep.METHOD_SETTINGS) == {
        "rkd", "stella", "cdm", "dskd", "emo", "talas", "geoode"
    }
    assert set(sweep.PAIRS) == {
        "qwen3_0.6b_to_minilm_h384", "bge_m3_to_minilm_h768", "qwen3_4b_to_bert_base"
    }
    args = sweep.parse_args([])
    jobs = sweep.build_jobs(args, Path("/runs/x"), Path("/cache"), Path("/train.csv"))
    assert len(jobs) == 3 * 7 * 3
    # One directory per (pair, method, seed): a job never overwrites another's run.
    assert len({job["run_dir"] for job in jobs}) == len(jobs)
    # Pair-major, so one pair's teacher cache is built once and each pair
    # finishes as a unit.
    assert [job["pair"] for job in jobs] == sorted(
        (job["pair"] for job in jobs), key=list(sweep.PAIRS).index
    )


def test_every_job_matches_the_notebooks_command(sweep, notebook_settings):
    """The whole 63-job grid, flag for flag, against the notebook's builder."""
    cells = _notebook_cells()
    args = sweep.parse_args(["--dataset", notebook_settings["DATASET"]])
    train_data = REPO_ROOT / notebook_settings["DATASETS"][args.dataset]["path"]
    cache_dir = Path("/cache")
    run_root = Path("/runs/x")

    for pair_name, pair in notebook_settings["PAIRS"].items():
        namespace = {
            "sys": sys,
            "Path": Path,
            "PROJECT_DIR": REPO_ROOT,
            "RUN_ROOT": run_root,
            "TRAIN_DATA": train_data,
            "CACHE_DIR": cache_dir,
            "STUDENT_MODEL": pair["student"],
            "TEACHER_MODEL": pair["teacher"],
            "TEACHER_POOLING": pair["teacher_pooling"],
            "TEACHER_SPECIAL_TOKEN": pair["teacher_special_token"],
            "EMO_TEACHER_SPECIAL_TOKEN": pair["emo_teacher_special_token"],
            **{
                key: notebook_settings[key]
                for key in ("EPOCHS", "MAX_LENGTH", "NUM_WORKERS", "EVAL_EVERY",
                            "HOLD_OUT_VALIDATION", "EVAL_RETRIEVAL",
                            "GEOODE_H0_BATCH_SIZE")
            },
        }
        # Everything above `JOBS = [` is the notebook's build_command.
        exec(compile(cells["commands"].split("JOBS = [")[0], "<notebook-commands>", "exec"),
             namespace)
        notebook_build = namespace["build_command"]

        for method, settings in notebook_settings["METHOD_SETTINGS"].items():
            for seed in notebook_settings["SEEDS"]:
                run_dir = run_root / pair_name / method / f"seed_{seed}"
                mine = sweep.build_command(
                    args, pair_name, method, seed, run_dir, cache_dir, train_data
                )
                # The notebook lays runs out as RUN_ROOT/<method>/seed_<seed>; the
                # sweep inserts the pair. That directory is the only difference.
                theirs = [
                    str(run_dir) if part == str(run_root / method / f"seed_{seed}") else part
                    for part in notebook_build(method, settings, seed)
                ]
                assert mine == theirs, f"{pair_name} {method} seed={seed}"


def test_the_geoode_arm_carries_the_notebooks_extra_flags(sweep):
    args = sweep.parse_args([])
    command = sweep.build_command(
        args, "qwen3_4b_to_bert_base", "geoode", 42,
        Path("/runs/x"), Path("/cache"), Path("/train.csv"),
    )
    for flag, value in [("--lambda_topo", "1.0"), ("--lambda_ctr", "0.0"),
                        ("--gauge_refit_every", "1")]:
        assert command[command.index(flag) + 1] == value
    # Cached methods read the shared teacher cache; the online ones never see it.
    assert "--cache_dir" in command
    online = sweep.build_command(
        args, "qwen3_4b_to_bert_base", "cdm", 42,
        Path("/runs/x"), Path("/cache"), Path("/train.csv"),
    )
    assert "--cache_dir" not in online


def test_the_per_pair_sub_word_markers_reach_the_methods_that_read_them(sweep):
    args = sweep.parse_args([])
    build = lambda pair, method: sweep.build_command(  # noqa: E731
        args, pair, method, 42, Path("/runs/x"), Path("/cache"), Path("/train.csv")
    )
    # BGE-M3 is SentencePiece and its BOS differs from Qwen3's, so cdm and emo
    # need different markers for the same corpus.
    bge_cdm = build("bge_m3_to_minilm_h768", "cdm")
    assert bge_cdm[bge_cdm.index("--teacher_special_token") + 1] == "▁"
    bge_emo = build("bge_m3_to_minilm_h768", "emo")
    assert bge_emo[bge_emo.index("--teacher_special_token") + 1] == "<s>"
    qwen_cdm = build("qwen3_0.6b_to_minilm_h384", "cdm")
    assert qwen_cdm[qwen_cdm.index("--teacher_special_token") + 1] == "Ġ"
    # Qwen3 pairs set no EMO marker, so emo keeps the config's.
    assert "--teacher_special_token" not in build("qwen3_0.6b_to_minilm_h384", "emo")
    # Nothing else reads a marker.
    for method in ("rkd", "stella", "dskd", "talas", "geoode"):
        assert "--teacher_special_token" not in build("bge_m3_to_minilm_h768", method)


def test_the_commands_parse_as_main_py_arguments(sweep):
    """Every flag the sweep emits is a flag main.py actually has."""
    from unittest.mock import patch

    from main import get_config, parse_args as main_parse_args

    args = sweep.parse_args([])
    for pair_name in sweep.PAIRS:
        for method in sweep.METHOD_SETTINGS:
            command = sweep.build_command(
                args, pair_name, method, 42, Path("/runs/x"), Path("/cache"),
                Path("/train.csv"),
            )
            with patch.object(sys, "argv", ["main.py", *command[2:]]):
                parsed = main_parse_args()
            assert parsed.method == method
            config = get_config(method, parsed)
            assert config.batch_size == sweep.METHOD_SETTINGS[method]["batch_size"]
            assert config.learning_rate == sweep.METHOD_SETTINGS[method]["learning_rate"]
            assert config.seed == 42


def test_seeds_must_be_distinct_but_a_single_seed_is_allowed(sweep):
    """A repeated seed is two copies of one run wearing the std of a real pair, so it
    stays an error. One seed is not: it is a smoke run, or a single setting checked
    before paying for the full grid, and it simply has no std column."""
    assert sweep.parse_args(["--seeds", "42"]).seeds == [42]
    with pytest.raises(SystemExit):
        sweep.parse_args(["--seeds", "42", "42", "43"])


def test_a_single_seed_table_prints_the_mean_without_a_std(sweep):
    import numpy as np

    assert sweep.mean_sd(76.25, np.nan) == "76.25"
    assert sweep.mean_sd(76.25, 0.4) == "76.25 ± 0.40"


def test_the_sweep_is_sequential_unless_asked_otherwise(sweep):
    """Co-location is opt-in: the default run is the one the efficiency table needs."""
    args = sweep.parse_args([])
    assert args.max_parallel == 1
    assert args.gpus == [sweep.CUDA_VISIBLE_DEVICES]

    args = sweep.parse_args(["--max-parallel", "4", "--gpus", "0", "1"])
    assert sweep.job_runner.gpu_slots(args.gpus, args.max_parallel) == [
        "0", "0", "0", "0", "1", "1", "1", "1"
    ]
    with pytest.raises(SystemExit):
        sweep.parse_args(["--max-parallel", "0"])


def test_the_warm_up_is_the_job_command_with_cache_only_and_no_save_dir(sweep, monkeypatch):
    """One teacher pass per pair before the fan-out, not one per slot.

    The warm-up has to be the job's own command, or it would build a cache under a
    different key than the jobs then look for.
    """
    args = sweep.parse_args(["--max-parallel", "2"])
    jobs = sweep.build_jobs(args, Path("/runs/x"), Path("/cache"), Path("/train.csv"))
    calls = []
    monkeypatch.setattr(sweep.subprocess, "run", lambda command, **kwargs: calls.append(command))

    sweep.prewarm_caches(args, jobs)

    # One per pair, and only for the methods that read a cache at all.
    assert len(calls) == len(sweep.PAIRS)
    for command, pair_name in zip(calls, sweep.PAIRS):
        assert command[-1] == "--cache_only"
        assert "--save_dir" not in command
        assert "--cache_dir" in command
        assert pair_name in ("qwen3_0.6b_to_minilm_h384", "bge_m3_to_minilm_h768",
                             "qwen3_4b_to_bert_base")

    uncached = [job for job in jobs if job["method"] in ("cdm", "dskd", "emo")]
    calls.clear()
    sweep.prewarm_caches(args, uncached)
    assert calls == []
