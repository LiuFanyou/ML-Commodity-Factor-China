from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class SpatioTemporalBlock(nn.Module):
    def __init__(self, hidden: int, attention_dim: int, kernel: int, dropout: float):
        super().__init__()
        self.q = nn.Linear(hidden, attention_dim, bias=False)
        self.k = nn.Linear(hidden, attention_dim, bias=False)
        # Learnable spectral graph filter g_theta(Lambda), implemented as a
        # second-order polynomial in the normalized-Laplacian eigenvalues.
        self.spectral_coeff = nn.Parameter(torch.empty(3, hidden))
        nn.init.xavier_uniform_(self.spectral_coeff)
        self.spatial = nn.Linear(hidden, hidden)
        self.temporal = nn.Conv1d(hidden, hidden * 2, kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(attention_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, nodes, time, hidden]
        summary = x.mean(dim=2)
        adjacency = torch.softmax(self.q(summary) @ self.k(summary).transpose(-1, -2) / self.scale, dim=-1)
        # Paper Sec. IV-D(a): second-order polynomial spectral graph filter.
        # For g(Lambda) = theta_0 + theta_1 Lambda + theta_2 Lambda^2,
        # U g(Lambda) U^T X is exactly theta_0 X + theta_1 L X +
        # theta_2 L^2 X. Applying the polynomial directly avoids a batched
        # eigendecomposition and keeps gradients through the dynamic graph.
        graph = 0.5 * (adjacency + adjacency.transpose(-1, -2))
        degree = graph.sum(dim=-1).clamp_min(1e-6)
        d_inv_sqrt = degree.rsqrt()
        normalized = d_inv_sqrt.unsqueeze(-1) * graph * d_inv_sqrt.unsqueeze(-2)
        eye = torch.eye(graph.shape[-1], device=x.device, dtype=x.dtype).expand_as(graph)
        laplacian = eye - normalized
        laplacian_x = torch.einsum("bij,bjth->bith", laplacian, x)
        laplacian2_x = torch.einsum("bij,bjth->bith", laplacian, laplacian_x)
        spatial = (
            x * self.spectral_coeff[0]
            + laplacian_x * self.spectral_coeff[1]
            + laplacian2_x * self.spectral_coeff[2]
        )
        graph_message = torch.einsum("bij,bjth->bith", adjacency, x)
        spatial = self.spatial(spatial + graph_message)
        b, n, t, h = spatial.shape
        # Paper Sec. IV-D(b): DFT -> Conv1D + GLU -> IDFT. Real and imaginary
        # parts share the same temporal filter, preserving a real output.
        frequency = torch.fft.rfft(spatial, dim=2, norm="ortho")
        freq_len = frequency.shape[2]
        components = torch.stack((frequency.real, frequency.imag), dim=0)
        components = components.permute(0, 1, 2, 4, 3).reshape(2 * b * n, h, freq_len)
        components = F.glu(self.temporal(components), dim=1)
        components = components.reshape(2, b, n, h, freq_len).permute(0, 1, 2, 4, 3)
        filtered_frequency = torch.complex(components[0], components[1])
        temporal = torch.fft.irfft(filtered_frequency, n=t, dim=2, norm="ortho")
        return self.norm(x + self.dropout(temporal)), adjacency


class HCGNNDaily(nn.Module):
    tasks = ("cpd", "gap", "ma", "price")

    def __init__(
        self, n_features: int, hidden_dim: int = 64, attention_dim: int = 32,
        temporal_kernel: int = 3, num_blocks: int = 2, dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(hidden_dim, attention_dim, temporal_kernel, dropout)
            for _ in range(num_blocks)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * num_blocks, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = nn.ModuleDict({task: nn.Linear(hidden_dim, 1) for task in self.tasks})

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        outputs, adjacency = [], None
        for block in self.blocks:
            h, adjacency = block(h)
            outputs.append(h[:, :, -1])
        return self.readout(torch.cat(outputs, dim=-1)), adjacency

    def forward(self, x: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, adjacency = self.encode(x)
        return self.heads[task](z).squeeze(-1), z, adjacency

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        for name, param in self.named_parameters():
            if not name.startswith("heads."):
                yield param


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1.flatten(0, 1), dim=-1)
    z2 = F.normalize(z2.flatten(0, 1), dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
