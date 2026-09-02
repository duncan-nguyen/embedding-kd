"""Where the pair-classification threshold is calibrated.

Only the pair family carries a decision threshold across splits, so this is the one
place a test score can quietly stop being held out.
"""

import pytest

import distiller as distiller_module
from config import BaseConfig
from distiller import KnowledgeDistiller

PAIR_METRICS = {
    "best_threshold": 0.5,
    "accuracy": 0.8,
    "f1": 0.78,
    "precision": 0.79,
    "recall": 0.77,
    "average_precision": 0.85,
}


@pytest.fixture
def evaluator(monkeypatch):
    """A distiller with only the pieces `evaluate` touches, and fake benchmarks."""
    calls = {}

    def fake_classification(model, tasks, tokenizer):
        return {path: {"accuracy": 0.7, "f1": 0.68} for _, path in tasks}

    def fake_pair(model, tasks, tokenizer, thresholds=None):
        calls["thresholds"] = thresholds
        return (
            {path: dict(PAIR_METRICS) for path in tasks},
            {index: 0.42 for index in range(len(tasks))},
        )

    def fake_sts(model, tasks, tokenizer):
        return dict.fromkeys(tasks, 0.6)

    def fake_retrieval(model, tasks, tokenizer):
        calls["retrieval"] = list(tasks)
        return {
            path: {"ndcg_at_10": 0.3, "recall_at_10": 0.4, "mrr_at_10": 0.25}
            for path in tasks
        }

    monkeypatch.setattr(distiller_module, "eval_classification_task", fake_classification)
    monkeypatch.setattr(distiller_module, "eval_pair_task", fake_pair)
    monkeypatch.setattr(distiller_module, "eval_sts_task", fake_sts)
    monkeypatch.setattr(distiller_module, "eval_retrieval_task", fake_retrieval)

    instance = object.__new__(KnowledgeDistiller)
    instance.config = BaseConfig()
    instance.model_student = None
    instance.tok_student = None
    instance.current_epoch = 0
    return instance, calls


def test_test_split_reuses_validation_thresholds_under_the_held_out_protocol(evaluator):
    instance, calls = evaluator
    instance.config.pair_threshold_source = "validation"

    instance.evaluate("validation")
    assert calls["thresholds"] is None  # validation sweeps its own

    results = instance.evaluate("test")

    assert calls["thresholds"] == {0: 0.42, 1: 0.42, 2: 0.42}
    assert results["pair_threshold_source"] == "validation"


def test_test_split_without_validation_is_refused_when_held_out(evaluator):
    instance, _ = evaluator
    instance.config.pair_threshold_source = "validation"

    with pytest.raises(RuntimeError, match="thresholds selected on validation"):
        instance.evaluate("test")


def test_calibrating_on_test_sweeps_the_test_split(evaluator):
    instance, calls = evaluator
    instance.config.pair_threshold_source = "test"

    results = instance.evaluate("test")

    # No thresholds passed in means eval_pair_task sweeps on the split it scores.
    assert calls["thresholds"] is None
    assert results["pair_threshold_source"] == "test"


def test_calibrating_on_test_does_not_need_a_validation_pass(evaluator):
    instance, _ = evaluator
    instance.config.pair_threshold_source = "test"

    assert not hasattr(instance, "pair_validation_thresholds")
    assert instance.evaluate("test")["pair_threshold_source"] == "test"


def test_calibrating_on_test_leaves_validation_untouched(evaluator):
    instance, calls = evaluator
    instance.config.pair_threshold_source = "test"

    results = instance.evaluate("validation")

    assert calls["thresholds"] is None
    assert results["pair_threshold_source"] == "test"
    assert instance.pair_validation_thresholds == {0: 0.42, 1: 0.42, 2: 0.42}


def test_unknown_threshold_source_is_rejected(evaluator):
    instance, _ = evaluator
    instance.config.pair_threshold_source = "train"

    with pytest.raises(ValueError, match="pair_threshold_source"):
        instance.evaluate("test")


def test_cli_selects_the_threshold_source(monkeypatch):
    import sys

    from main import get_config, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "geoode", "--pair_threshold_source", "validation"],
    )
    config = get_config("geoode", parse_args())

    assert config.pair_threshold_source == "validation"
    # Naming the held-out source asks for the validation pass that selects it.
    assert config.evaluate_test_each_epoch is False
    assert BaseConfig().pair_threshold_source == "test"


def test_per_epoch_test_evaluation_requires_test_thresholds():
    """The contradiction has to surface before the models load, not after epoch 1."""
    config = BaseConfig()
    config.evaluate_test_each_epoch = True
    config.pair_threshold_source = "validation"

    with pytest.raises(ValueError, match="pair_threshold_source='test'"):
        KnowledgeDistiller._validate_eval_config(config)


def test_per_epoch_test_evaluation_accepts_test_thresholds():
    config = BaseConfig()
    config.evaluate_test_each_epoch = True
    config.pair_threshold_source = "test"

    KnowledgeDistiller._validate_eval_config(config)  # must not raise


def test_default_eval_config_is_test_only():
    """One evaluation pass per run, on the test split, with nothing held out."""
    config = BaseConfig()

    assert config.evaluate_test_each_epoch is True
    assert config.pair_threshold_source == "test"
    KnowledgeDistiller._validate_eval_config(config)


def test_bad_threshold_source_is_rejected_before_training():
    config = BaseConfig()
    config.pair_threshold_source = "train"

    with pytest.raises(ValueError, match="pair_threshold_source"):
        KnowledgeDistiller._validate_eval_config(config)


def test_test_evaluation_is_titled_by_epoch_until_the_final_one(evaluator, capsys):
    instance, _ = evaluator
    instance.config.pair_threshold_source = "test"
    instance.current_epoch = 2

    instance.evaluate("test")
    assert "TEST - EPOCH 3" in capsys.readouterr().out

    instance.evaluate("test", final=True)
    assert "FINAL TEST" in capsys.readouterr().out


def test_cli_flag_carries_the_threshold_source(monkeypatch):
    import sys

    from main import get_config, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "geoode", "--evaluate_test_each_epoch"],
    )
    config = get_config("geoode", parse_args())

    assert config.evaluate_test_each_epoch is True
    # Implied, because the flag exists precisely to run without a validation pass.
    assert config.pair_threshold_source == "test"
    KnowledgeDistiller._validate_eval_config(config)


def test_cli_opt_out_restores_the_held_out_protocol(monkeypatch):
    """The one flag turns both halves of the protocol back on."""
    import sys

    from main import get_config, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "geoode", "--no-evaluate_test_each_epoch"],
    )
    config = get_config("geoode", parse_args())

    assert config.evaluate_test_each_epoch is False
    assert config.pair_threshold_source == "validation"
    KnowledgeDistiller._validate_eval_config(config)


def test_explicit_threshold_source_still_wins_and_is_rejected_if_contradictory(monkeypatch):
    import sys

    from main import get_config, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--method",
            "geoode",
            "--evaluate_test_each_epoch",
            "--pair_threshold_source",
            "validation",
        ],
    )
    config = get_config("geoode", parse_args())

    assert config.pair_threshold_source == "validation"
    with pytest.raises(ValueError, match="pair_threshold_source='test'"):
        KnowledgeDistiller._validate_eval_config(config)


def test_final_test_runs_validation_once_when_no_epoch_eval_happened(evaluator):
    # eval_every=0 skips every per-epoch validation; the final test still has to
    # calibrate on validation, so the distiller runs that pass itself.
    instance, calls = evaluator
    instance.config.save_dir = None
    instance.config.pair_threshold_source = "validation"

    assert not hasattr(instance, "pair_validation_thresholds")
    instance._ensure_pair_thresholds()
    assert instance.pair_validation_thresholds == {0: 0.42, 1: 0.42, 2: 0.42}

    results = instance.evaluate("test")
    assert calls["thresholds"] == {0: 0.42, 1: 0.42, 2: 0.42}
    assert results["pair_threshold_source"] == "validation"


def test_final_test_skips_the_extra_validation_when_calibrating_on_test(evaluator):
    instance, calls = evaluator
    instance.config.pair_threshold_source = "test"

    instance._ensure_pair_thresholds()

    assert "thresholds" not in calls  # no validation pass was run
    assert not hasattr(instance, "pair_validation_thresholds")


def test_retrieval_runs_on_test_only_and_stays_out_of_the_sentence_averages(evaluator):
    """Retrieval joins AVG (ALL) but must not move AVG (IOD)/AVG (OOD).

    The sentence-level groups are what earlier runs are reported against, so
    folding three nDCG@10 scores into AVG (OOD) would silently redefine a number
    the paper already carries.
    """
    instance, calls = evaluator
    instance.config.pair_threshold_source = "test"

    validation = instance.evaluate("validation")
    assert "retrieval" not in calls  # no qrels on the validation side
    assert validation["retrieval"] == {}
    assert "avg_retrieval" not in validation["summary"]

    test = instance.evaluate("test")
    assert calls["retrieval"] == distiller_module.test_retrieval_tasks
    assert test["summary"]["avg_retrieval"] == pytest.approx(0.3)
    assert test["summary"]["avg_iod"] == pytest.approx(validation["summary"]["avg_iod"])
    assert test["summary"]["avg_ood"] == pytest.approx(validation["summary"]["avg_ood"])
    assert test["summary"]["avg_all"] < validation["summary"]["avg_all"]


def test_retrieval_can_be_switched_off(evaluator):
    instance, calls = evaluator
    instance.config.pair_threshold_source = "test"
    instance.config.eval_retrieval = False

    results = instance.evaluate("test")

    assert "retrieval" not in calls
    assert results["retrieval"] == {}
    assert "avg_retrieval" not in results["summary"]


def test_cli_disables_retrieval(monkeypatch):
    import sys

    from main import get_config, parse_args

    monkeypatch.setattr(
        sys, "argv", ["main.py", "--method", "geoode", "--no_eval_retrieval"]
    )
    config = get_config("geoode", parse_args())

    assert config.eval_retrieval is False
    assert BaseConfig().eval_retrieval is True
