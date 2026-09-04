from scripts.ablations import run_decomposition as decomposition


def _value(command, flag):
    return command[command.index(flag) + 1]


def test_default_decomposition_is_15k_and_three_seeds():
    args = decomposition.parse_args([])
    jobs = decomposition.build_jobs(args)

    assert args.train_data == "data/train_set/merged_3_data_5k_each.csv"
    assert args.seeds == [42, 43, 44]
    assert len(decomposition.ARMS) == 6
    assert len(jobs) == 18
    assert len({job["name"] for job in jobs}) == 18


def test_refit_ablation_is_binary_and_changes_only_refitting():
    args = decomposition.parse_args(["--seeds", "42"])
    jobs = decomposition.build_jobs(args)
    off = next(job for job in jobs if job["arm"] == "combined_original_refit_off")
    on = next(job for job in jobs if job["arm"] == "combined_original_refit_on")

    ignored = {"arm", "name", "refit", "run_dir", "log_path", "command"}
    assert {key: value for key, value in off.items() if key not in ignored} == {
        key: value for key, value in on.items() if key not in ignored
    }
    assert _value(off["command"], "--gauge_refit_every") == "0"
    assert _value(on["command"], "--gauge_refit_every") == "1"
    assert "--gauge_align" in off["command"]
    assert "--gauge_align" in on["command"]


def test_objective_and_source_flags_match_each_arm():
    args = decomposition.parse_args(["--seeds", "42"])

    for job in decomposition.build_jobs(args):
        command = job["command"]
        assert _value(command, "--lambda_end") == str(job["lambda_end"])
        assert _value(command, "--lambda_topo") == str(
            args.lambda_topo if job["uses_h0"] else 0.0
        )
        assert _value(command, "--topo_teacher_source") == job["h0_source"]
        assert "--no_eval_retrieval" in command
