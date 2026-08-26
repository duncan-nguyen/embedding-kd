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
    # The paper fixes alpha and the two main loss weights and tunes beta / lambda_ctr.
    assert config.alpha == 1.0
    assert config.lambda_end == 1.0
    assert config.lambda_dyn == 1.0
    assert config.guidance_schedule == "linear"


def test_geoode_flags_override_the_config():
    config = _config(
        "--method",
        "geoode",
        "--beta",
        "2.5",
        "--lambda_ctr",
        "0.05",
        "--guidance_schedule",
        "power",
        "--guidance_power",
        "2.0",
        "--student_pooling",
        "mean",
    )

    assert config.beta == 2.5
    assert config.lambda_ctr == 0.05
    assert config.guidance_schedule == "power"
    assert config.guidance_power == 2.0
    assert config.student_pooling == "mean"


def test_pca_subtract_mean_flag_is_tri_state():
    # Unset leaves the config default alone; the two spellings set it either way.
    assert _config("--method", "geoode").pca_subtract_mean is GeoODEConfig.pca_subtract_mean
    assert _config("--method", "geoode", "--pca_subtract_mean").pca_subtract_mean is True
    assert _config("--method", "geoode", "--no-pca_subtract_mean").pca_subtract_mean is False


def test_gauge_align_flag_is_tri_state():
    assert _config("--method", "geoode").gauge_align is GeoODEConfig.gauge_align
    assert _config("--method", "geoode", "--no-gauge_align").gauge_align is False
    assert _config("--method", "geoode", "--gauge_align").gauge_align is True


def test_gauge_refit_flag_overrides_the_config():
    assert _config("--method", "geoode").gauge_refit_every == GeoODEConfig.gauge_refit_every
    assert _config("--method", "geoode", "--gauge_refit_every", "1").gauge_refit_every == 1


def test_geoode_flags_are_rejected_for_other_methods():
    with pytest.raises(ValueError, match="only supported by the geoode method"):
        _config("--method", "talas", "--beta", "2.0")


def test_geoode_config_round_trips_through_to_dict():
    config = _config("--method", "geoode", "--lambda_dyn", "0.5")

    values = config.to_dict()

    assert values["distill_method"] == "geoode"
    assert values["lambda_dyn"] == 0.5
    assert values["contrastive_view"] == "dropout"
