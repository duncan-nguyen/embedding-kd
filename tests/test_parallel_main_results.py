"""Exercise the real subprocess queue without CUDA or model downloads."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sweep():
    path = Path(__file__).resolve().parents[1] / "scripts/experiments/run_main_results.py"
    spec = importlib.util.spec_from_file_location("parallel_sweep_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER = """
import json, os, pathlib, sys, time
r = pathlib.Path(sys.argv[1])
delay = float(sys.argv[2])
fail = sys.argv[3] == 'fail'
r.joinpath('observed.json').write_text(json.dumps({
    'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'started': time.time(), 'pid': os.getpid()
}))
time.sleep(delay)
if fail:
    raise SystemExit(9)
r.joinpath('metrics.jsonl').write_text(json.dumps({'test': {'score': 1}}) + '\\n')
r.joinpath('finished.txt').write_text(str(time.time()))
"""


def job(tmp_path, method, seed, delay=0.05, fail=False, pair="qwen3_0.6b_to_minilm_h384"):
    r = tmp_path / pair / method / f"seed_{seed}"
    return {"pair": pair, "method": method, "seed": seed, "run_dir": r,
            "command": [sys.executable, "-c", WORKER, str(r), str(delay),
                        "fail" if fail else "ok"]}


def test_two_devices_are_isolated_and_completed_jobs_are_skipped(sweep, tmp_path):
    args = sweep.parse_args(["--jobs-per-gpu", "1", "--cuda-visible-devices", "2,3"])
    done = job(tmp_path, "rkd", 42)
    done["run_dir"].mkdir(parents=True)
    (done["run_dir"] / "metrics.jsonl").write_text('{"test":{"score":1}}\n')
    jobs = [done, job(tmp_path, "stella", 42), job(tmp_path, "stella", 43)]
    result = sweep.run_jobs(args, jobs, tmp_path)
    assert list(result.status).count("complete") == 2
    assert list(result.status).count("skipped_complete") == 1
    assert not (done["run_dir"] / "observed.json").exists()
    observed = [json.loads((j["run_dir"] / "observed.json").read_text()) for j in jobs[1:]]
    assert {x["gpu"] for x in observed} == {"2", "3"}
    assert abs(observed[0]["started"] - observed[1]["started"]) < 1
    state = json.loads((tmp_path / "scheduler_state.json").read_text())
    assert not state["active"] and state["pending"] == 0


def test_a_free_slot_refills_while_another_job_is_still_running(sweep, tmp_path):
    args = sweep.parse_args(["--jobs-per-gpu", "2", "--cuda-visible-devices", "2"])
    jobs = [job(tmp_path, "stella", 42, delay=2.5),
            job(tmp_path, "stella", 43), job(tmp_path, "stella", 44)]
    sweep.run_jobs(args, jobs, tmp_path)
    later = json.loads((jobs[2]["run_dir"] / "observed.json").read_text())
    assert later["started"] < float((jobs[0]["run_dir"] / "finished.txt").read_text())
    timing = sweep.read_timing(jobs[0]["run_dir"])
    assert timing["concurrent_on_same_gpu"] is True
    assert timing["cuda_visible_devices"] == "2"


def test_first_cache_builder_finishes_before_another_reader_starts(sweep, tmp_path):
    args = sweep.parse_args(["--jobs-per-gpu", "1", "--cuda-visible-devices", "2,3"])
    jobs = [job(tmp_path, "rkd", 42), job(tmp_path, "rkd", 43)]
    sweep.run_jobs(args, jobs, tmp_path)
    second = json.loads((jobs[1]["run_dir"] / "observed.json").read_text())
    assert second["started"] >= float((jobs[0]["run_dir"] / "finished.txt").read_text())


def test_retry_keeps_the_unfinished_attempt_and_keep_going_records_failure(sweep, tmp_path):
    args = sweep.parse_args(["--jobs-per-gpu", "1", "--cuda-visible-devices", "2",
                             "--retry-unfinished", "--keep-going"])
    first = job(tmp_path, "stella", 42, fail=True)
    first["run_dir"].mkdir(parents=True)
    (first["run_dir"] / "metrics.jsonl").write_text('{"train":{"loss":2}}\n')
    result = sweep.run_jobs(args, [first, job(tmp_path, "stella", 43)], tmp_path)
    assert list(result.status) == ["failed", "complete"]
    assert sweep.read_timing(first["run_dir"])["returncode"] == 9
    assert len(list(first["run_dir"].parent.glob("seed_42.stale-*"))) == 1


def test_fail_fast_drains_active_jobs_and_leaves_queued_jobs_unstarted(sweep, tmp_path):
    args = sweep.parse_args(["--jobs-per-gpu", "1", "--cuda-visible-devices", "2,3"])
    jobs = [job(tmp_path, "stella", 42, fail=True),
            job(tmp_path, "stella", 43, delay=1.3), job(tmp_path, "stella", 44)]
    with pytest.raises(RuntimeError, match="Parallel job failed"):
        sweep.run_jobs(args, jobs, tmp_path)
    assert (jobs[1]["run_dir"] / "finished.txt").exists()
    assert not (jobs[2]["run_dir"] / "observed.json").exists()


def test_4b_emo_is_exclusive_but_another_gpu_can_work(sweep, tmp_path):
    heavy = job(tmp_path, "emo", 42, pair="qwen3_4b_to_bert_base")
    light = job(tmp_path, "rkd", 42)
    active = [{"gpu": "2", "job": heavy}]
    assert sweep.next_parallel_job([light], active, "2", set()) is None
    assert sweep.next_parallel_job([light], active, "3", set()) == light
    assert sweep.next_parallel_job([heavy], [{"gpu": "2", "job": light}], "2", set()) is None


def test_4b_online_teacher_can_share_with_cached_student_only(sweep, tmp_path):
    large = job(tmp_path, "stella", 42, pair="qwen3_4b_to_bert_base")
    small = job(tmp_path, "stella", 42)
    cached = job(tmp_path, "rkd", 42)
    active = [{"gpu": "2", "job": large}]
    assert sweep.next_parallel_job([small, cached], active, "2", set()) == cached
    assert sweep.next_parallel_job([large], [{"gpu": "2", "job": small}], "2", set()) is None


def test_runtime_slot_override_controls_dispatch(sweep, tmp_path):
    args = sweep.parse_args(["--jobs-per-gpu", "2", "--cuda-visible-devices", "2"])
    (tmp_path / "parallel_limits.json").write_text('{"2":1}')
    jobs = [job(tmp_path, "stella", 42), job(tmp_path, "stella", 43)]
    sweep.run_jobs(args, jobs, tmp_path)
    second = json.loads((jobs[1]["run_dir"] / "observed.json").read_text())
    assert second["started"] >= float((jobs[0]["run_dir"] / "finished.txt").read_text())
    assert sweep.read_timing(jobs[0]["run_dir"])["concurrent_on_same_gpu"] is False
    assert json.loads((tmp_path / "scheduler_state.json").read_text())["slot_limits"] == {"2": 1}


def test_failed_attempt_is_prioritized_on_a_free_gpu(sweep, tmp_path):
    retry = job(tmp_path, "geoode", 42, pair="qwen3_4b_to_bert_base")
    retry["retry_failed"] = True
    assert sweep.next_parallel_job([job(tmp_path, "emo", 42), retry], [], "1", set()) == retry


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux worker handoff")
def test_handoff_keeps_original_worker_and_starts_new_gpu(sweep, tmp_path):
    import datetime
    import subprocess

    running = job(tmp_path, "emo", 42, delay=1.5)
    running["run_dir"].mkdir(parents=True)
    command = running["command"] + ["--save_dir", str(running["run_dir"])]
    with subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2"},
                          stdout=subprocess.DEVNULL, start_new_session=True) as worker:
        entry = {**sweep._job_keys(running), "gpu": "2", "pid": worker.pid,
                 "start_ticks": sweep.process_identity(worker.pid)[1],
                 "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        handoff = tmp_path / "handoff.json"
        handoff.write_text(json.dumps({"active": [entry]}))
        args = sweep.parse_args(["--jobs-per-gpu", "1", "--cuda-visible-devices", "1,2",
                                 "--adopt-running", str(handoff)])
        pending = job(tmp_path, "stella", 43)
        result = sweep.run_jobs(args, [running, pending], tmp_path)
        assert list(result.status) == ["complete", "complete"]
        original = json.loads((running["run_dir"] / "observed.json").read_text())
        added = json.loads((pending["run_dir"] / "observed.json").read_text())
        assert original["pid"] == worker.pid and original["gpu"] == "2"
        assert added["gpu"] == "1"
        timing = sweep.read_timing(running["run_dir"])
        assert timing["adopted"] and timing["returncode"] is None
        assert timing["completion_evidence"] == "final_test_record"


def test_adopted_pid_reuse_is_treated_as_an_exited_worker(sweep, tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "process_identity", lambda pid: ("S", "different-start"))
    process = sweep.AdoptedProcess({"pid": 123, "start_ticks": "original-start"}, tmp_path)
    assert process.poll() == 1


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux worker handoff")
def test_adoption_rejects_an_unrelated_output_directory(sweep, tmp_path):
    entry = {"pid": os.getpid(), "start_ticks": sweep.process_identity(os.getpid())[1]}
    with pytest.raises(RuntimeError, match="does not belong"):
        sweep.AdoptedProcess(entry, tmp_path)


def test_drain_marker_does_not_start_more_jobs(sweep, tmp_path):
    args = sweep.parse_args(["--jobs-per-gpu", "1"])
    (tmp_path / "DRAIN").touch()
    pending = job(tmp_path, "stella", 42)
    sweep.run_jobs(args, [pending], tmp_path)
    assert not pending["run_dir"].exists()
    state = json.loads((tmp_path / "scheduler_state.json").read_text())
    assert state["draining"] and state["pending"] == 1


def test_sigterm_cleans_up_worker_process_groups(sweep, tmp_path):
    import signal
    import threading
    import time

    args = sweep.parse_args(["--jobs-per-gpu", "1", "--cuda-visible-devices", "2"])
    pending = job(tmp_path, "stella", 42, delay=60)
    observed = pending["run_dir"] / "observed.json"
    def stop_when_started():
        deadline = time.monotonic() + 5
        while not observed.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)
    thread = threading.Thread(target=stop_when_started)
    thread.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            sweep.run_jobs(args, [pending], tmp_path)
    finally:
        thread.join()
    pid = json.loads(observed.read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("argv", [
    ["--jobs-per-gpu", "-1"],
    ["--jobs-per-gpu", "1", "--cuda-visible-devices", "2,2"],
    ["--jobs-per-gpu", "1", "--cuda-visible-devices", "2,"],
])
def test_invalid_parallel_configuration_is_rejected(sweep, argv):
    with pytest.raises(SystemExit):
        sweep.parse_args(argv)
