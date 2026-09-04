"""The parallel job runner: slots, concurrency, resume-safe failure handling.

The runner exists so a sweep can fill a card that one job leaves mostly empty,
which means the properties worth pinning are the ones a sweep depends on: no more
jobs than slots ever run at once, every job is told which GPU it owns, a failure
stops the queue without killing what is already running, and the rows come back in
the order the jobs were given.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import job_runner  # noqa: E402


def _job(tmp_path, name, script):
    """A job whose command is a python one-liner, so no GPU is involved."""
    return {
        "name": name,
        "command": [sys.executable, "-c", script],
        "log_path": tmp_path / f"{name}.log",
    }


def test_gpu_slots_repeats_each_device():
    assert job_runner.gpu_slots(["0"], 4) == ["0", "0", "0", "0"]
    assert job_runner.gpu_slots([0, 1], 2) == ["0", "0", "1", "1"]
    # An entry may name several devices, for a job that wants more than one card.
    assert job_runner.gpu_slots(["0,1"]) == ["0,1"]
    with pytest.raises(ValueError):
        job_runner.gpu_slots([], 1)
    with pytest.raises(ValueError):
        job_runner.gpu_slots(["0"], 0)


def test_every_job_runs_and_rows_come_back_in_order(tmp_path):
    jobs = [_job(tmp_path, f"job{i}", f"print({i})") for i in range(5)]
    rows = job_runner.run_jobs_parallel(
        jobs, cwd=REPO_ROOT, slots=["0", "0"], poll_seconds=1e6, printer=lambda _: None
    )
    assert [row["name"] for row in rows] == [f"job{i}" for i in range(5)]
    assert {row["status"] for row in rows} == {"complete"}
    assert [Path(row["log_path"]).read_text().strip() for row in rows] == [
        str(i) for i in range(5)
    ]


def test_no_more_than_one_job_per_slot_runs_at_once(tmp_path):
    """Each job records the interval it was alive; two slots means at most two
    intervals overlap at any instant."""
    script = (
        "import json, os, time, sys;"
        "start=time.time();"
        "time.sleep(0.5);"
        "open(sys.argv[1],'w').write(json.dumps"
        "([start, time.time(), os.environ['CUDA_VISIBLE_DEVICES']]))"
    )
    jobs = []
    for index in range(6):
        job = _job(tmp_path, f"job{index}", script)
        job["command"].append(str(tmp_path / f"span{index}.json"))
        jobs.append(job)

    rows = job_runner.run_jobs_parallel(
        jobs,
        cwd=REPO_ROOT,
        slots=["0", "1"],
        poll_seconds=1e6,
        printer=lambda _: None,
    )
    assert {row["status"] for row in rows} == {"complete"}

    spans = [json.loads((tmp_path / f"span{i}.json").read_text()) for i in range(6)]
    edges = sorted(
        [(start, +1) for start, _, _ in spans] + [(end, -1) for _, end, _ in spans]
    )
    live = peak = 0
    for _, delta in edges:
        live += delta
        peak = max(peak, live)
    # Two slots: never three at once, and -- the point of the exercise -- really two.
    assert peak == 2
    assert {gpu for _, _, gpu in spans} == {"0", "1"}
    assert {row["gpu"] for row in rows} == {"0", "1"}


def test_a_failure_stops_the_queue_but_reports_what_ran(tmp_path):
    jobs = [
        _job(tmp_path, "ok", "print('fine')"),
        _job(tmp_path, "boom", "raise SystemExit(3)"),
        *[_job(tmp_path, f"later{i}", "print('later')") for i in range(3)],
    ]
    rows = job_runner.run_jobs_parallel(
        jobs, cwd=REPO_ROOT, slots=["0"], poll_seconds=1e6, printer=lambda _: None
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["ok"]["status"] == "complete"
    assert by_name["boom"]["status"] == "failed"
    assert by_name["boom"]["returncode"] == 3
    assert [by_name[f"later{i}"]["status"] for i in range(3)] == ["not_started"] * 3


def test_keep_going_runs_the_rest(tmp_path):
    jobs = [
        _job(tmp_path, "boom", "raise SystemExit(1)"),
        _job(tmp_path, "after", "print('after')"),
    ]
    rows = job_runner.run_jobs_parallel(
        jobs,
        cwd=REPO_ROOT,
        slots=["0"],
        stop_on_error=False,
        poll_seconds=1e6,
        printer=lambda _: None,
    )
    assert [row["status"] for row in rows] == ["failed", "complete"]


def test_on_finish_may_downgrade_a_zero_exit(tmp_path):
    """What a caller needs to call an exit-zero run that left no final test a
    failure -- the resume rule both runners share."""
    jobs = [_job(tmp_path, "silent", "print('no metrics written')")]
    rows = job_runner.run_jobs_parallel(
        jobs,
        cwd=REPO_ROOT,
        slots=["0"],
        stop_on_error=False,
        poll_seconds=1e6,
        printer=lambda _: None,
        on_finish=lambda job, row: {**row, "status": "failed"},
    )
    assert rows[0]["status"] == "failed"


def test_tail_progress_reads_through_carriage_returns(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Epoch 1\nLoss 0.5\rEpoch 2:  50%|#####     | 5/10\r", encoding="utf-8")
    assert job_runner.tail_progress(log) == "Epoch 2:  50%|#####     | 5/10"
    assert job_runner.tail_progress(tmp_path / "missing.log") == ""


def test_threads_are_divided_between_co_located_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(job_runner.os, "cpu_count", lambda: 16)
    assert job_runner.cpu_threads_per_job(4) == 4
    assert job_runner.cpu_threads_per_job(32) == 1

    job = _job(tmp_path, "threads", "import os; print(os.environ['OMP_NUM_THREADS'])")
    job_runner.run_jobs_parallel(
        [job], cwd=REPO_ROOT, env={}, slots=["0", "0", "0", "0"],
        poll_seconds=1e6, printer=lambda _: None,
    )
    assert Path(job["log_path"]).read_text().strip() == "4"
