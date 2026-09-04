"""Run training jobs as subprocesses -- one at a time, or several at once.

One job here is one ``main.py``: a student, a corpus, a seed. At the batch sizes
the paper uses it holds a handful of gigabytes of a card that has ninety, and it
is the corpus length rather than the card that sets how long it takes, so a sweep
run one job at a time leaves most of the GPU idle for hours. The fix is to put
more jobs on the card, not to make one job bigger: a larger batch would change
the experiment, a second process does not. Every job is its own seeded process
reading its own dataloader, and computes exactly what it would have computed
alone.

What co-location *does* change is every number that is a rate. Jobs sharing a
card interleave on the same SMs, so ms/step and samples/s read slower and the
free-memory headroom reads smaller than the same run alone would. Anything that
ends up in an efficiency table has to be measured with one job on the card; see
``notebooks/00_main_results.ipynb`` cell 7, which refuses to publish timings it
knows were taken under co-location.

The scheduler is deliberately dumb: a fixed list of slots, one job per free slot,
a slot is a string handed to the child as ``CUDA_VISIBLE_DEVICES``. That covers
"four jobs on one big card" (``["0"] * 4``), "one job per card"
(``["0", "1"]``) and "one job that wants both cards" (``["0,1"]``) without the
scheduler having to know anything about GPUs.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

# tqdm rewrites its bar in place with a carriage return, so the last progress the
# log holds is the tail of the file, not its last newline-terminated line.
PROGRESS_TAIL_BYTES = 8192
TICK_SECONDS = 1.0
TERMINATE_GRACE_SECONDS = 30.0


def gpu_slots(gpus: Sequence[str | int], jobs_per_gpu: int = 1) -> list[str]:
    """One slot per concurrent job: what that job will see as its GPU.

    ``gpus`` entries are passed through as written, so an entry may itself name
    several devices (``"0,1"``) for a job meant to use more than one card.
    """
    if jobs_per_gpu < 1:
        raise ValueError(f"jobs_per_gpu must be >= 1, got {jobs_per_gpu}")
    slots = [str(gpu) for gpu in gpus for _ in range(jobs_per_gpu)]
    if not slots:
        raise ValueError("gpus must name at least one device")
    return slots


def tail_progress(log_path: str | Path, max_bytes: int = PROGRESS_TAIL_BYTES) -> str:
    """The last thing a running job printed, progress bars included."""
    path = Path(log_path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            blob = handle.read()
    except OSError:
        return ""
    text = blob.decode("utf-8", errors="replace")
    for line in reversed(text.replace("\r", "\n").split("\n")):
        if line.strip():
            return line.strip()
    return ""


def cpu_threads_per_job(slots: int) -> int:
    """Torch's intra-op threads for one of ``slots`` co-located jobs.

    Left at its default every job would open one thread per core and the jobs
    would spend their time preempting each other in the collate and the optimizer
    rather than in the GPU queue.
    """
    cores = os.cpu_count() or 1
    return max(1, cores // max(1, slots))


def _terminate(entries: list[dict]) -> None:
    for entry in entries:
        process = entry["process"]
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    for entry in entries:
        process = entry["process"]
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for entry in entries:
        entry["handle"].close()


def run_jobs_parallel(
    jobs: Sequence[dict],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    slots: Sequence[str] = ("0",),
    poll_seconds: float = 30.0,
    stop_on_error: bool = True,
    on_start: Callable[[dict, str], None] | None = None,
    on_finish: Callable[[dict, dict], dict | None] | None = None,
    printer: Callable[[str], None] = print,
) -> list[dict]:
    """Run every job, at most ``len(slots)`` of them at a time.

    Each job is a dict with a ``name``, a ``command`` (argv) and a ``log_path``;
    its other keys are carried into the returned row untouched. Output is written
    straight to the log rather than streamed to the caller -- with several jobs
    talking at once an interleaved stdout is unreadable -- and every
    ``poll_seconds`` one status line per running job is printed instead.

    ``on_finish(job, row)`` is where the caller decides what a finished job is
    worth: it may return a replacement row, which is what lets a caller downgrade
    an exit-zero job that left no final-test record to ``failed``. When a job ends
    up anything other than ``complete`` and ``stop_on_error`` is set, no further
    job is launched, the jobs already running are left to finish, and the rest
    come back as ``not_started``.

    Resume is the caller's business: filter the completed runs out of ``jobs``
    before calling, exactly as the sequential runner does.
    """
    slots = list(slots)
    if not slots:
        raise ValueError("run_jobs_parallel needs at least one slot")

    base_env = dict(os.environ if env is None else env)
    base_env.setdefault("OMP_NUM_THREADS", str(cpu_threads_per_job(len(slots))))

    pending = list(enumerate(jobs))
    running: list[dict] = []
    free = list(slots)
    rows: dict[int, dict] = {}
    stopping = False
    last_status = 0.0
    total = len(jobs)

    try:
        while pending or running:
            while pending and free and not stopping:
                index, job = pending.pop(0)
                gpu = free.pop(0)
                log_path = Path(job["log_path"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    job["command"],
                    cwd=str(cwd),
                    env={**base_env, "CUDA_VISIBLE_DEVICES": gpu},
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append(
                    {
                        "index": index,
                        "job": job,
                        "gpu": gpu,
                        "process": process,
                        "handle": handle,
                        "log_path": log_path,
                        "started": time.perf_counter(),
                    }
                )
                printer(
                    f"[START {index + 1}/{total}] {job['name']} on GPU {gpu} "
                    f"(pid {process.pid}) -> {log_path}"
                )
                if on_start is not None:
                    on_start(job, gpu)

            if not running:
                break

            time.sleep(min(TICK_SECONDS, poll_seconds))

            for entry in [e for e in running if e["process"].poll() is not None]:
                running.remove(entry)
                free.append(entry["gpu"])
                entry["handle"].close()
                seconds = time.perf_counter() - entry["started"]
                returncode = entry["process"].returncode
                row = {
                    **entry["job"],
                    "gpu": entry["gpu"],
                    "returncode": returncode,
                    "seconds": seconds,
                    "status": "complete" if returncode == 0 else "failed",
                }
                if on_finish is not None:
                    row = on_finish(entry["job"], row) or row
                rows[entry["index"]] = row
                printer(
                    f"[{row['status'].upper()}] {entry['job']['name']} "
                    f"in {seconds / 60:.1f} min (rc={returncode})"
                )
                if row["status"] != "complete" and stop_on_error:
                    stopping = True
                    printer(
                        f"[STOP] {entry['job']['name']} failed; no new job will be "
                        f"launched. {len(running)} still running, {len(pending)} queued."
                    )

            now = time.perf_counter()
            if running and now - last_status >= poll_seconds:
                last_status = now
                for entry in running:
                    minutes = (now - entry["started"]) / 60
                    printer(
                        f"  [{minutes:6.1f} min] gpu {entry['gpu']} "
                        f"{entry['job']['name']}: {tail_progress(entry['log_path'])}"
                    )

            if stopping and not running:
                break
    except BaseException:
        # Ctrl-C in a notebook kills this loop, not the children: without this the
        # jobs keep running and keep holding the card.
        printer(f"[ABORT] terminating {len(running)} running job(s)")
        _terminate(running)
        raise

    for index, job in pending:
        rows[index] = {
            **job,
            "gpu": None,
            "returncode": None,
            "seconds": 0.0,
            "status": "not_started",
        }
    return [rows[index] for index in range(total)]
