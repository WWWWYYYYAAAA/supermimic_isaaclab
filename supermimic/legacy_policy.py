"""Inference-only adapter for pre-v7 BatchNorm policies."""
import torch
from torch import nn
from .normalization import FeatureBatchNorm, HistoryBatchNorm
from .networks import mlp, run_mlp
from .mdp import NUM_J

class LegacyPolicy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, hist: int = 10, z_dim: int = 32):
        super().__init__()
        self.hist = hist
        self.goal_dim = goal_dim
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, [2048, 1024, 512, 256], NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = HistoryBatchNorm(obs_dim, hist)
        self.goal_norm = FeatureBatchNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist * obs_dim), flattened by frames, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist)
        return run_mlp(self.dec, torch.cat((z, o), -1))
