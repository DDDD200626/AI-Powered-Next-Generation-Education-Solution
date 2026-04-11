from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture
def augment_fn():
    from edu_tools.team_torch_model import _augment_training_features

    return _augment_training_features


def test_augment_noop_when_all_disabled(monkeypatch, augment_fn) -> None:
    monkeypatch.setenv("TEAM_SEMANTIC_ENCODER", "0")
    monkeypatch.setenv("TEAM_SEMANTIC_AUTO_REG", "0")
    monkeypatch.delenv("TEAM_SEMANTIC_ANTI_OVERFIT", raising=False)
    monkeypatch.delenv("TEAM_SEMANTIC_TRAIN_DROPOUT", raising=False)
    monkeypatch.delenv("TEAM_SEMANTIC_TRAIN_SCALE", raising=False)
    monkeypatch.setenv("TEAM_TORCH_INPUT_NOISE_STD", "0")
    x = torch.arange(34, dtype=torch.float32).unsqueeze(0)
    y = augment_fn(x)
    assert y is x


def test_augment_scales_semantic_slice(monkeypatch, augment_fn) -> None:
    monkeypatch.setenv("TEAM_SEMANTIC_TRAIN_SCALE", "0.5")
    monkeypatch.setenv("TEAM_SEMANTIC_TRAIN_DROPOUT", "0")
    monkeypatch.setenv("TEAM_TORCH_INPUT_NOISE_STD", "0")
    x = torch.ones(2, 34)
    y = augment_fn(x)
    assert torch.allclose(y[:, :26], torch.ones(2, 26))
    assert torch.allclose(y[:, 26:], torch.full((2, 8), 0.5))


def test_augment_drop_all_semantic_when_p_one(monkeypatch, augment_fn) -> None:
    monkeypatch.setenv("TEAM_SEMANTIC_TRAIN_DROPOUT", "1")
    monkeypatch.setenv("TEAM_SEMANTIC_TRAIN_SCALE", "1")
    monkeypatch.setenv("TEAM_TORCH_INPUT_NOISE_STD", "0")
    x = torch.ones(1, 34)
    y = augment_fn(x)
    assert y[0, 26:].abs().max().item() == 0.0
    assert y[0, :26].mean().item() == 1.0


def test_anti_overfit_sets_defaults(monkeypatch) -> None:
    monkeypatch.delenv("TEAM_SEMANTIC_TRAIN_DROPOUT", raising=False)
    monkeypatch.delenv("TEAM_SEMANTIC_TRAIN_SCALE", raising=False)
    monkeypatch.setenv("TEAM_SEMANTIC_ANTI_OVERFIT", "1")
    from edu_tools.team_torch_model import _semantic_train_dropout_p, _semantic_train_scale

    assert _semantic_train_dropout_p() == pytest.approx(0.18)
    assert _semantic_train_scale() == pytest.approx(0.92)


def test_auto_reg_when_encoder_on_without_explicit_dropout(monkeypatch) -> None:
    monkeypatch.setenv("TEAM_SEMANTIC_ENCODER", "1")
    monkeypatch.delenv("TEAM_SEMANTIC_TRAIN_DROPOUT", raising=False)
    monkeypatch.delenv("TEAM_SEMANTIC_TRAIN_SCALE", raising=False)
    monkeypatch.delenv("TEAM_SEMANTIC_ANTI_OVERFIT", raising=False)
    monkeypatch.setenv("TEAM_SEMANTIC_AUTO_REG", "1")
    from edu_tools.team_torch_model import _semantic_train_dropout_p, _semantic_train_scale

    assert _semantic_train_dropout_p() == pytest.approx(0.15)
    assert _semantic_train_scale() == pytest.approx(0.94)
