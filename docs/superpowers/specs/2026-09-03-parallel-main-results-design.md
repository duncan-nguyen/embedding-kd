# Parallel main-results sweep

The H200 run has two assigned GPUs (physical IDs 2 and 3), but putting the
student on one and teacher on the other leaves substantial idle time. Cached
teacher methods release the teacher entirely. The objective is total sweep
throughput while preserving every model, seed, batch size, epoch, and loss.

## Design and implementation plan

- [x] Inspect the runner, resume boundaries, cache writes, and live GPU usage.
- [x] Select independent jobs per GPU over the existing model split or DDP.
- [x] Add an opt-in `--jobs-per-gpu` queue; zero retains sequential behavior.
- [x] Assign each subprocess exactly one GPU; refill free slots dynamically.
  Begin with two slots per GPU and compare live throughput before increasing.
  EMO with the 4B teacher runs alone on its GPU to bound memory use.
  Allow per-device slot tuning through `parallel_limits.json` without restarting
  active jobs. Other 4B online jobs share only with cached student jobs.
- [x] Skip completed results and preserve unfinished attempts before retrying.
  Serialize the first cached job of a pair until it completes, then allow
  readers to share the completed cache. Only one scheduler writes status.
- [x] Preserve job logs and timing records; publish active jobs and device
  placement in `scheduler_state.json`. Record whether timing included another
  job on the same GPU, since it is not an isolated efficiency measurement.
- [x] Test GPU isolation, slot refill, cache initialization exclusion, resume,
  error handling, and cleanup with small subprocesses; retain notebook parity.
- [x] Deploy using the server project's existing `.venv`, replace the serial
  controller without losing completed jobs, and verify both GPU queues live.

Pending jobs prioritize the slower online methods, with a different method
preferred when sharing a GPU. This reduces the long tail while overlapping
different CPU/GPU demands. Workers keep the experiment's original commands.
No packages are installed globally. A drain marker stops new dispatches and
lets active jobs finish for subsequent reconfiguration.

GPU utilization cannot remain at 100% during data loading, CPU losses,
checkpoint writes, and evaluation. Success means higher measured total
throughput and automatic refill, not a promised utilization percentage.

The user authorized immediate implementation after specification; no further
approval gate applies. Self-review: scope is limited to orchestration, with
unchanged training semantics and explicit accounting for shared timing.

Live inspection also found STELLA ignored `eval_every=0` in stage 2. Make its
evaluation cadence obey the existing flag, matching single-stage training;
keep the unconditional final test and all five training epochs. Verify zero,
one, and two as evaluation intervals with a lightweight training harness.

Validation: 20 scheduler/notebook parity tests and 26 save/evaluation tests
passed in the remote project virtual environment. The serial STELLA seed 42
finished before the transition at 2026-09-03 10:19:35 UTC. Four completed jobs
were retained. Initial two-slot placement ran EMO plus GeoODE on each GPU;
an asymmetric two/three-slot trial then measured throughput and utilization.

Final live limit: two slots per GPU for the online-method queue. The three-slot
trial reached roughly 90–99% GPU utilization, but EMO allocator reservations
grew to about 64 GiB in individual processes, so additional slots were not kept
for this phase. Already-running third jobs drain naturally. At 10:26:42 UTC,
eight jobs were complete with no recorded failures; five jobs were still active
during this drain, and the instantaneous GPU utilizations were 100% and 92%.

## Add GPU 1 without restarting training

The user additionally authorized physical GPU 1. Preserve the four live EMO
workers on GPUs 2 and 3 by handing their PIDs, Linux start ticks, output paths,
and original start times to a replacement controller. Terminate only the old
controller, whose workers already have separate sessions and direct log files.
Validate process identities, then dispatch new work on GPU 1, prioritizing the
failed 4B GeoODE attempt. A non-child exit status cannot be recovered reliably;
adopted workers use final-test records for completion and record a null exit code.
Test adoption without relaunch, PID-reuse protection, retry priority, and the
existing queue behavior before deployment.

Deployed at 2026-09-03 11:30:01 UTC after 24 scheduler/parity tests passed.
The four original EMO PIDs (3261492, 3261494, 3262221, 3262998) continued on GPUs
2 and 3. GPU 1 received the failed 4B GeoODE seed 42 retry and BGE-M3 EMO seed 43.
The new controller reported eight completed jobs, six running, and 49 pending.
