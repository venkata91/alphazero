"""Tests for the shared device-resolution helper."""
from __future__ import annotations

import torch

from alphazero.device import resolve_device


def test_explicit_cpu_returns_cpu_device():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_resolves_consistently_for_trainer_and_supervised():
    """Trainer._resolve_device and supervised._resolve_device must agree
    on every input — they share the same helper, but verify the bridges."""
    from alphazero.supervised import _resolve_device as sup_resolve
    from alphazero.trainer import Trainer

    for name in ("auto", "cpu"):
        assert Trainer._resolve_device(name) == sup_resolve(name), (
            f"Trainer and supervised disagree on '{name}' device"
        )


def test_auto_picks_cuda_if_available(monkeypatch):
    """CUDA wins over MPS when both are present (matches Trainer's original order)."""

    class _Patch:
        @staticmethod
        def is_available():
            return True

    class _MPSPatch:
        is_available = staticmethod(lambda: True)

    monkeypatch.setattr(torch.cuda, "is_available", _Patch.is_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", _MPSPatch.is_available)

    # 'auto' should prefer CUDA when both available
    assert resolve_device("auto") == torch.device("cuda")
