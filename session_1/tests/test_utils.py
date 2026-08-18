from __future__ import annotations

from pathlib import Path

from src.utils import PROJECT_ROOT, get_feature_target, load_config, load_data, resolve_path


def test_resolve_path_relative_is_rooted_at_project_root():
    resolved = resolve_path("configs/config.yaml")
    assert resolved == PROJECT_ROOT / "configs" / "config.yaml"
    assert resolved.exists()


def test_resolve_path_absolute_is_unchanged(tmp_path):
    absolute = tmp_path / "file.txt"
    assert resolve_path(absolute) == absolute


def test_load_config_has_expected_sections():
    config = load_config()
    assert {"data", "model", "artifacts", "api"} <= config.keys()
    assert config["data"]["target_column"] == "final_grade"


def test_load_data_reads_expected_columns(real_config):
    df = load_data(real_config)
    assert len(df) > 0
    assert "final_grade" in df.columns


def test_get_feature_target_drops_target_and_leak_columns(real_config, real_features_target):
    X, y = real_features_target
    assert "final_grade" not in X.columns
    for leak_col in real_config["data"]["drop_columns"]:
        assert leak_col not in X.columns
    assert len(X) == len(y)
