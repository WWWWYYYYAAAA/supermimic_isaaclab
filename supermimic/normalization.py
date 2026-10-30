"""Policy BatchNorm with explicit collection updates and flat inference inputs."""

import torch
from torch import nn
from torch.nn import functional as F


class FeatureBatchNorm(nn.BatchNorm1d):
    """Use native BatchNorm statistics; forward always uses saved statistics.

    Forward expects FP32 inputs. Only update() observes new environment data.
    Policy optimization, imagined rollouts and inference must use the same
    normalization, even in train mode.
    """

    def __init__(self, dim: int):
        super().__init__(dim, eps=1e-4, momentum=0.1, affine=False)

    @torch.jit.unused
    @torch.no_grad()
    def update(self, x: torch.Tensor):
        samples = x.detach().float().reshape(-1, self.num_features)
        # Native training BatchNorm needs at least two samples per feature.
        if samples.size(0) < 2:
            return
        previous_mode = self.training
        self.train(True)
        try:
            super().forward(samples)
        finally:
            self.train(previous_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit inference mode avoids native BatchNorm's runtime shape/mode
        # branches in TorchScript while retaining the native batch_norm op.
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight=None,
            bias=None,
            training=False,
            momentum=0.0,
            eps=self.eps,
        )


class HistoryBatchNorm(FeatureBatchNorm):
    """Normalize pre-flattened [B, H*D] history with shared per-feature stats."""

    def __init__(self, dim: int, history: int):
        super().__init__(dim)
        self.history = history
        # Derived buffers avoid allocating repeats in every policy forward.
        # Keep the checkpoint's original D-wide BatchNorm statistics unchanged.
        self.register_buffer("history_mean", self.running_mean.repeat(history), persistent=False)
        self.register_buffer("history_var", self.running_var.repeat(history), persistent=False)

    def _refresh_history(self):
        self.history_mean.copy_(self.running_mean.repeat(self.history))
        self.history_var.copy_(self.running_var.repeat(self.history))

    @torch.jit.unused
    @torch.no_grad()
    def update(self, x: torch.Tensor):
        super().update(x)
        self._refresh_history()

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._refresh_history()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            x, self.history_mean, self.history_var,
            weight=None, bias=None, training=False, momentum=0.0, eps=self.eps,
        )
