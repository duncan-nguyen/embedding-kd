from scripts.experiments import run_sensitivity as sensitivity


def test_default_plan_is_the_15k_three_seed_protocol():
    args = sensitivity.parse_args([])
    specs = sensitivity.build_specs(args)
    jobs = sensitivity.build_jobs(args, specs)

    assert args.train_data == "data/train_set/merged_3_data_5k_each.csv"
    assert args.seeds == [42, 43, 44]
    assert len(specs) == 10
    assert len(jobs) == 30
    assert len({job["name"] for job in jobs}) == 30


def test_sensitivity_changes_one_factor_at_a_time():
    args = sensitivity.parse_args([])

    for spec in sensitivity.build_specs(args):
        changed = [
            key
            for key, default in sensitivity.DEFAULTS.items()
            if spec[key] != default
        ]
        assert len(changed) == (0 if spec["sweep"] == "default" else 1)
        if changed:
            assert changed == [spec["sweep"]]


def test_generated_commands_preserve_the_main_gate_protocol():
    args = sensitivity.parse_args([])

    for job in sensitivity.build_jobs(args, sensitivity.build_specs(args)):
        command = job["command"]

        def value(flag):
            return command[command.index(flag) + 1]

        assert value("--train_data").endswith("merged_3_data_5k_each.csv")
        assert value("--seed") in {"42", "43", "44"}
        assert value("--projection_type") == "pca"
        assert value("--gauge_rotation") == "procrustes"
        assert value("--gauge_refit_every") == "1"
        assert value("--lambda_end") == "1.0"
        assert value("--lambda_ctr") == "0.0"
        assert value("--lambda_h1") == "0.0"
        assert "--no_eval_retrieval" in command
        assert "--no_wandb" in command

        assert value("--topo_batch_size") == value("--batch_size")
