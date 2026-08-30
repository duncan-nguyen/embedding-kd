"""CLI and config wiring for the geoode method."""

import sys

import pytest

from config import GeoODEConfig
from main import get_config, parse_args


def _config(*argv):
    original = sys.argv
    sys.argv = ["main.py", *argv]
    try:
        args = parse_args()
        return get_config(args.method, args)
    finally:
        sys.argv = original


def test_geoode_method_selects_its_config():
    config = _config("--method", "geoode")

    assert isinstance(config, GeoODEConfig)
    assert config.distill_method == "geoode"
    # The objective is L_end + L_ctr and nothing else: the endpoint term at weight 1
    # plus the contrastive regulariser.
    assert config.lambda_end == 1.0
    assert config.lambda_ctr == 0.5


def test_geoode_flags_override_the_config():
    config = _config(
        "--method",
        "geoode",
        "--lambda_end",
        "0.7",
        "--lambda_ctr",
        "0.05",
        "--student_pooling",
        "mean",
    )

    assert config.lambda_end == 0.7
    assert config.lambda_ctr == 0.05
    assert config.student_pooling == "mean"


def test_pca_subtract_mean_flag_is_tri_state():
    # Unset leaves the config default alone; the two spellings set it either way.
    assert _config("--method", "geoode").pca_subtract_mean is GeoODEConfig.pca_subtract_mean
    assert _config("--method", "geoode", "--pca_subtract_mean").pca_subtract_mean is True
    assert _config("--method", "geoode", "--no-pca_subtract_mean").pca_subtract_mean is False


def test_projection_type_flag_selects_the_subspace_arm():
    assert _config("--method", "geoode").projection_type == "pca"
    for arm in ("random", "random_gaussian", "mrl_prefix", "learned_t2s", "learned_s2t"):
        assert _config("--method", "geoode", "--projection_type", arm).projection_type == arm


def test_learned_projector_lr_scale_flag_overrides_the_config():
    # The learned arms are baselines, so their one knob has to be sweepable from
    # the command line rather than edited into the config.
    assert _config("--method", "geoode").learned_projector_lr_scale == GeoODEConfig.learned_projector_lr_scale
    config = _config("--method", "geoode", "--projection_type", "learned_t2s", "--learned_projector_lr_scale", "5")
    assert config.learned_projector_lr_scale == 5.0
    with pytest.raises(ValueError, match="only supported by the geoode method"):
        _config("--method", "rkd", "--learned_projector_lr_scale", "5")


def test_projection_seed_flag_overrides_the_config():
    assert _config("--method", "geoode").projection_seed == GeoODEConfig.projection_seed
    assert _config("--method", "geoode", "--projection_seed", "7").projection_seed == 7


def test_pca_center_fit_flag_is_tri_state():
    assert _config("--method", "geoode").pca_center_fit is GeoODEConfig.pca_center_fit
    assert _config("--method", "geoode", "--pca_center_fit").pca_center_fit is True
    # The uncentered-SVD ablation.
    assert _config("--method", "geoode", "--no-pca_center_fit").pca_center_fit is False


def test_gauge_rotation_flag_selects_the_orientation_arm():
    assert _config("--method", "geoode").gauge_rotation == "procrustes"
    config = _config("--method", "geoode", "--gauge_rotation", "random")
    assert config.gauge_rotation == "random"
    # The random gauge is a control *for* the alignment, so it still applies one.
    assert config.gauge_align is True


def test_gauge_random_seed_flag_overrides_the_config():
    assert _config("--method", "geoode").gauge_random_seed == GeoODEConfig.gauge_random_seed
    assert _config("--method", "geoode", "--gauge_random_seed", "4").gauge_random_seed == 4


def test_gauge_align_flag_is_tri_state():
    assert _config("--method", "geoode").gauge_align is GeoODEConfig.gauge_align
    assert _config("--method", "geoode", "--no-gauge_align").gauge_align is False
    assert _config("--method", "geoode", "--gauge_align").gauge_align is True


def test_gauge_refit_flag_overrides_the_config():
    assert _config("--method", "geoode").gauge_refit_every == GeoODEConfig.gauge_refit_every
    assert _config("--method", "geoode", "--gauge_refit_every", "1").gauge_refit_every == 1


def test_geoode_flags_are_rejected_for_other_methods():
    with pytest.raises(ValueError, match="only supported by the geoode method"):
        _config("--method", "talas", "--lambda_end", "2.0")


def test_geoode_config_round_trips_through_to_dict():
    config = _config("--method", "geoode", "--lambda_ctr", "0.5")

    values = config.to_dict()

    assert values["distill_method"] == "geoode"
    assert values["lambda_ctr"] == 0.5
    assert values["contrastive_view"] == "dropout"


def test_endpoint_loss_flag_selects_the_mse_baseline():
    assert _config("--method", "geoode").endpoint_loss == "cosine"
    assert _config("--method", "geoode", "--endpoint_loss", "mse").endpoint_loss == "mse"


def test_pca_mse_baseline_is_expressible_from_the_cli():
    # PCA target + MSE, no gauge, no contrastive term: the sentence-transformers
    # <= v5.4 distillation recipe, run through the same code path as the recipe.
    config = _config(
        "--method", "geoode",
        "--endpoint_loss", "mse",
        "--no-gauge_align",
        "--lambda_ctr", "0",
    )
    assert config.endpoint_loss == "mse"
    assert config.gauge_align is False
    assert config.lambda_ctr == 0.0
    assert config.projection_type == "pca"


def test_h0_flags_default_off_and_are_forwarded():
    # The H0 term is a control, so the recipe run must be bit-identical without it.
    default = _config("--method", "geoode")
    assert default.lambda_topo == 0.0
    assert default.topo_metric == "chord"

    config = _config(
        "--method", "geoode", "--lambda_topo", "0.1", "--topo_metric", "angular"
    )
    assert config.lambda_topo == 0.1
    assert config.topo_metric == "angular"
