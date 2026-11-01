# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.


from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape: int | tuple[int, ...] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        """Return the current running mean."""
        return self._mean.squeeze(0).clone()  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        x = x.detach().reshape((-1,) + self._mean.shape[1:]).float()
        if x.shape[0] == 0:
            return
        count_x = torch.tensor(x.shape[0], dtype=self.count.dtype, device=self.count.device)  # type: ignore
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)

        if torch.distributed.is_initialized():
            # Compute the global mean first, then combine the local variances around that mean
            local_mean_x = mean_x
            torch.distributed.all_reduce(count_x)
            mean_sum_x = mean_x * x.shape[0]
            torch.distributed.all_reduce(mean_sum_x)
            mean_x = mean_sum_x / count_x
            var_sum_x = x.shape[0] * (var_x + (local_mean_x - mean_x).square())
            torch.distributed.all_reduce(var_sum_x)
            var_x = var_sum_x / count_x

        self.count += count_x
        rate = count_x / self.count
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean



# Legacy checkpoint inference support. New networks use EmpiricalNormalization.
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
