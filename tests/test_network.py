import torch

from alphazero.network import ResidualBlock


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
