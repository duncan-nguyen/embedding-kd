"""CLI and config wiring for the rkd and simcse baselines."""

import sys

import pytest

from config import RKDConfig, SimCSEConfig
from main import get_config, parse_args


def _config(*argv):
    original = sys.argv
    sys.argv = ["main.py", *argv]
    try:
        args = parse_args()
        return get_config(args.method, args)
    finally:
        sys.argv = original


def test_rkd_method_selects_its_config():
    config = _config("--method", "rkd")

    assert isinstance(config, RKDConfig)
    assert config.distill_method == "rkd"
    # Park et al. (2019) weight the two relational terms 25 and 50.
    assert config.w_dist == 25.0
    assert config.w_angle == 50.0
    assert config.normalize_student is True
    # RKD trains on the same cached teacher embeddings as talas and geoode.
    assert config.cache_teacher is True


def test_rkd_flags_override_the_config():
    config = _config(
        "--method",
        "rkd",
        "--w_dist",
        "1.0",
        "--w_angle",
        "2.0",
        "--no-normalize_student",
        "--student_pooling",
        "mean",
    )

    assert config.w_dist == 1.0
    assert config.w_angle == 2.0
    assert config.normalize_student is False
    assert config.student_pooling == "mean"


def test_simcse_method_selects_its_config():
    config = _config("--method", "simcse")

    assert isinstance(config, SimCSEConfig)
    assert config.distill_method == "simcse"
    # Unsupervised SimCSE: two dropout views at tau = 0.05.
    assert config.simcse_view == "dropout"
    assert config.temperature == 0.05
    # There is no teacher embedding to profile the student's depth against.
    assert config.depth_log_every == 0


def test_simcse_view_flag_overrides_the_config():
    assert _config("--method", "simcse", "--simcse_view", "pair").simcse_view == "pair"


def test_normalize_student_flag_is_tri_state():
    assert _config("--method", "rkd").normalize_student is RKDConfig.normalize_student
    assert _config("--method", "rkd", "--normalize_student").normalize_student is True
    assert _config("--method", "rkd", "--no-normalize_student").normalize_student is False


def test_flags_are_rejected_by_the_methods_that_ignore_them():
    # A flag aimed at the wrong --method has to fail loudly: the run would
    # otherwise report a setting that never entered its objective.
    with pytest.raises(ValueError, match="only supported by the rkd method"):
        _config("--method", "geoode", "--w_dist", "1.0")
    with pytest.raises(ValueError, match="only supported by the simcse method"):
        _config("--method", "talas", "--simcse_view", "pair")
    with pytest.raises(
        ValueError, match="only supported by the geoode, rkd and simcse methods"
    ):
        _config("--method", "cdm", "--student_pooling", "mean")
