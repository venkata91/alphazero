from pathlib import Path

import pytest

from alphazero.config import TrainingConfig, load_config


def test_default_config_has_expected_values():
    cfg = TrainingConfig()
    assert cfg.n_blocks == 4
    assert cfg.n_channels == 32
    assert cfg.num_simulations == 50
    assert cfg.c_puct == 1.5
    assert cfg.num_iterations == 50
    assert cfg.batch_size == 64
    assert cfg.weight_decay == 1e-4


def test_config_is_frozen():
    cfg = TrainingConfig()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        cfg.n_blocks = 999  # type: ignore[misc]


def test_load_config_overrides_defaults(tmp_path: Path):
    p = tmp_path / "tiny.toml"
    p.write_text("n_blocks = 6\nnum_simulations = 100\n")
    cfg = load_config(p)
    assert cfg.n_blocks == 6
    assert cfg.num_simulations == 100
    assert cfg.batch_size == 64  # default preserved


def test_load_config_rejects_unknown_keys(tmp_path: Path):
    p = tmp_path / "bad.toml"
    p.write_text("not_a_real_key = 99\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)
