from model import FNOScore
import torch

def test_fno_score():
    n_layers = 4
    hidden_channels = 1
    hidden_dim = 64
    n_modes = 32
    model = FNOScore(n_layers, hidden_channels, hidden_dim, n_modes, verbose=False)
    x = torch.randn(8, 1, 128)
    t = torch.rand(8, 1)
    out = model(x, t)


if __name__ == "__main__":
    test_fno_score()