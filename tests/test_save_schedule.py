import sys

import pytest

from distiller import should_save_epoch
from main import get_config, parse_args


def test_save_every_cli_overrides_config_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "talas", "--save_every", "3"],
    )

    args = parse_args()
    config = get_config(args.method, args)

    assert config.save_every == 3


def test_non_positive_save_every_is_rejected(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--method", "talas", "--save_every", "0"],
    )

    args = parse_args()
    with pytest.raises(ValueError, match="positive integer"):
        get_config(args.method, args)


def test_five_epoch_periodic_save_schedule_only_selects_epoch_three():
    selected = [
        epoch_index + 1
        for epoch_index in range(5)
        if should_save_epoch(epoch_index, save_every=3)
    ]

    assert selected == [3]


@pytest.mark.parametrize("eval_every, evaluations", [(0, 0), (1, 3), (2, 1)])
def test_stella_respects_evaluation_cadence_and_always_runs_final_test(eval_every, evaluations):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from distiller import KnowledgeDistiller

    trainer = KnowledgeDistiller.__new__(KnowledgeDistiller)
    trainer.config = SimpleNamespace(
        student_model_name="student", teacher_model_name="teacher",
        epochs_stage1=2, epochs_stage2=3, batch_size=128, learning_rate=1e-4,
        save_every=5, warmup_ratio=0.0, evaluate_test_each_epoch=True,
        eval_every=eval_every,
    )
    trainer.model_student = torch.nn.ModuleDict({
        name: torch.nn.Linear(2, 2) for name in ("backbone", "fc1", "fc2", "fc3", "fc4")
    })
    trainer.train_loader = [0, 1]
    trainer.last_epoch_metrics = {"loss": 0.5}
    trainer.train_epoch = Mock(return_value=0.5)
    trainer.log_experiment_record = Mock()
    trainer.save_checkpoint = Mock()
    trainer._run_evaluation = Mock(return_value={"score": 1})
    trainer._final_test_evaluation = Mock()

    trainer._train_stella()

    assert trainer.train_epoch.call_count == 5
    assert trainer._run_evaluation.call_count == evaluations
    trainer._final_test_evaluation.assert_called_once_with(extra={"stage": 2})
    stage2 = [call.args[0] for call in trainer.log_experiment_record.call_args_list
              if call.args[0]["stage"] == 2]
    assert sum(record["test"] is not None for record in stage2) == evaluations
