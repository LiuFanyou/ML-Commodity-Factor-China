import torch

from hcgnn.model import SpatioTemporalBlock


def test_polynomial_filter_matches_eigendecomposition():
    """The direct degree-2 polynomial must preserve the old forward result."""
    torch.manual_seed(7)
    batch, nodes, time, hidden = 3, 8, 5, 6
    x = torch.randn(batch, nodes, time, hidden, dtype=torch.float64)

    graph = torch.rand(batch, nodes, nodes, dtype=torch.float64)
    graph = 0.5 * (graph + graph.transpose(-1, -2))
    degree = graph.sum(dim=-1).clamp_min(1e-6)
    normalized = degree.rsqrt().unsqueeze(-1) * graph * degree.rsqrt().unsqueeze(-2)
    laplacian = torch.eye(nodes, dtype=torch.float64).expand_as(graph) - normalized

    coeff = torch.randn(3, hidden, dtype=torch.float64)
    laplacian_x = torch.einsum("bij,bjth->bith", laplacian, x)
    direct = (
        x * coeff[0]
        + laplacian_x * coeff[1]
        + torch.einsum("bij,bjth->bith", laplacian, laplacian_x) * coeff[2]
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    fourier_x = torch.einsum("bij,bjth->bith", eigenvectors.transpose(-1, -2), x)
    powers = torch.stack(
        (torch.ones_like(eigenvalues), eigenvalues, eigenvalues.square()), dim=-1
    )
    spectral_filter = torch.einsum("bnk,kh->bnh", powers, coeff)
    via_eigendecomposition = torch.einsum(
        "bij,bjth->bith", eigenvectors, fourier_x * spectral_filter.unsqueeze(2)
    )

    torch.testing.assert_close(direct, via_eigendecomposition, rtol=1e-9, atol=1e-9)


def test_spatiotemporal_block_has_no_eigendecomposition(monkeypatch):
    """The production forward path must not fall back to torch.linalg.eigh."""
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("torch.linalg.eigh should not be called")

    monkeypatch.setattr(torch.linalg, "eigh", fail_if_called)
    block = SpatioTemporalBlock(hidden=8, attention_dim=4, kernel=3, dropout=0.0)
    output, adjacency = block(torch.randn(2, 5, 7, 8))

    assert output.shape == (2, 5, 7, 8)
    assert adjacency.shape == (2, 5, 5)
    assert torch.isfinite(output).all()
