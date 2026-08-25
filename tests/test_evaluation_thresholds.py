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

    monkeypatch.setattr(distiller_module, "eval_classification_task", fake_classification)
    monkeypatch.setattr(distiller_module, "eval_pair_task", fake_pair)
    monkeypatch.setattr(distiller_module, "eval_sts_task", fake_sts)

    instance = object.__new__(KnowledgeDistiller)
    instance.config = BaseConfig()
    instance.model_student = None
    instance.tok_student = None
    instance.current_epoch = 0
    return instance, calls


def test_test_split_reuses_validation_thresholds_by_default(evaluator):
    instance, calls = evaluator

    instance.evaluate("validation")
    assert calls["thresholds"] is None  # validation sweeps its own

    results = instance.evaluate("test")

    assert calls["thresholds"] == {0: 0.42, 1: 0.42, 2: 0.42}
    assert results["pair_threshold_source"] == "validation"


def test_test_split_without_validation_is_refused_by_default(evaluator):
    instance, _ = evaluator

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
        ["main.py", "--method", "geoode", "--pair_threshold_source", "test"],
    )
    config = get_config("geoode", parse_args())

    assert config.pair_threshold_source == "test"
    assert BaseConfig().pair_threshold_source == "validation"
