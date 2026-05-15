import numpy as np
import pytest
import torch

from alphazero.replay_buffer import ReplayBuffer


def _make_tuple(seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((3, 3, 3)).astype(np.float32)
    p = rng.dirichlet(np.ones(9)).astype(np.float32)
    z = float(rng.uniform(-1, 1))
    return (s, p, z)


def test_buffer_starts_empty():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0


def test_buffer_add_increases_length():
    buf = ReplayBuffer(capacity=10)
    buf.add([_make_tuple(i) for i in range(5)])
    assert len(buf) == 5


def test_buffer_overflow_drops_oldest():
    buf = ReplayBuffer(capacity=3)
    buf.add([_make_tuple(0)])
    buf.add([_make_tuple(1)])
    buf.add([_make_tuple(2)])
    buf.add([_make_tuple(99)])
    assert len(buf) == 3


def test_buffer_sample_returns_batch_shapes():
    buf = ReplayBuffer(capacity=100)
    buf.add([_make_tuple(i) for i in range(30)])
    states, policies, values = buf.sample(8)
    assert isinstance(states, torch.Tensor)
    assert isinstance(policies, torch.Tensor)
    assert isinstance(values, torch.Tensor)
    assert states.shape == (8, 3, 3, 3)
    assert policies.shape == (8, 9)
    assert values.shape == (8,)
    assert states.dtype == torch.float32
    assert policies.dtype == torch.float32
    assert values.dtype == torch.float32


def test_buffer_sample_raises_when_insufficient():
    buf = ReplayBuffer(capacity=10)
    buf.add([_make_tuple(0), _make_tuple(1)])
    with pytest.raises(ValueError):
        buf.sample(8)
