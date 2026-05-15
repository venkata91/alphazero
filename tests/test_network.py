import pytest

import torch

from alphazero.network import AlphaZeroNet, ResidualBlock


def test_residual_block_preserves_shape():
    block = ResidualBlock(n_channels=8)
    x = torch.randn(4, 8, 3, 3)
    y = block(x)
    assert y.shape == x.shape


def test_residual_block_skip_connection_is_identity_when_F_is_zero():
    """If we manually zero the conv weights, output should be ReLU(x) — the skip path."""
    block = ResidualBlock(n_channels=8)
    # Zero out both conv layers' weights AND set net to eval mode so BN doesn't update
    for layer in [block.conv1, block.conv2]:
        torch.nn.init.zeros_(layer.weight)
    block.eval()
    x = torch.relu(torch.randn(2, 8, 3, 3))   # non-negative so ReLU is identity
    y = block(x)
    # F(x) is now zero (conv weights are zero, BN in eval mode passes through),
    # so y = ReLU(0 + x) = ReLU(x) = x (since x >= 0)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


def test_alphazero_net_forward_shapes():
    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=2, n_channels=8)
    x = torch.randn(4, 3, 3, 3)
    policy_logits, value = net(x)
    assert policy_logits.shape == (4, 9)
    assert value.shape == (4,)


def test_alphazero_net_value_in_tanh_range():
    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=8)
    x = torch.randn(16, 3, 3, 3) * 5.0
    _, value = net(x)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_alphazero_net_state_dict_roundtrip(tmp_path):
    net1 = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=8)
    path = tmp_path / "net.pt"
    torch.save(net1.state_dict(), path)
    net2 = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=8)
    net2.load_state_dict(torch.load(path))
    net1.eval()
    net2.eval()
    x = torch.randn(2, 3, 3, 3)
    p1, v1 = net1(x)
    p2, v2 = net2(x)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(v1, v2)


def test_alphazero_net_overfits_a_single_batch():
    """Sanity check: a tiny net should be able to overfit a 4-example synthetic batch."""
    torch.manual_seed(0)
    net = AlphaZeroNet(input_shape=(3, 3, 3), action_size=9, n_blocks=1, n_channels=8)
    opt = torch.optim.Adam(net.parameters(), lr=1e-1)
    x = torch.randn(4, 3, 3, 3)
    target_pi = torch.softmax(torch.randn(4, 9), dim=-1)
    target_z = torch.tensor([0.7, -0.5, 0.3, -0.9])
    initial_loss = None
    for step in range(500):
        opt.zero_grad()
        logits, v = net(x)
        log_softmax = torch.log_softmax(logits, dim=-1)
        pl = -(target_pi * log_softmax).sum(dim=-1).mean()
        vl = torch.nn.functional.mse_loss(v, target_z)
        loss = pl + vl
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < initial_loss * 0.7, (
        f"Net failed to overfit a tiny batch: started at {initial_loss:.3f}, ended at {loss.item():.3f}"
    )
