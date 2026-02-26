"""Tests for latent space interpolation methods."""

import torch
from breeding.interpolation import lerp, slerp, interpolation_strip


def test_lerp_endpoints():
    z1 = torch.randn(256)
    z2 = torch.randn(256)
    assert torch.allclose(lerp(z1, z2, 0.0), z1)
    assert torch.allclose(lerp(z1, z2, 1.0), z2)


def test_lerp_midpoint():
    z1 = torch.ones(256)
    z2 = torch.ones(256) * 3.0
    mid = lerp(z1, z2, 0.5)
    assert torch.allclose(mid, torch.ones(256) * 2.0)


def test_slerp_endpoints():
    z1 = torch.randn(256)
    z2 = torch.randn(256)
    assert torch.allclose(slerp(z1, z2, 0.0), z1, atol=1e-5)
    assert torch.allclose(slerp(z1, z2, 1.0), z2, atol=1e-5)


def test_slerp_parallel_vectors():
    """When vectors are nearly parallel, slerp should fall back to lerp."""
    z1 = torch.ones(256)
    z2 = torch.ones(256) * 1.001
    result = slerp(z1, z2, 0.5)
    assert result.shape == z1.shape
    assert torch.isfinite(result).all()


def test_interpolation_strip_length():
    z1 = torch.randn(256)
    z2 = torch.randn(256)
    strip = interpolation_strip(z1, z2, steps=10)
    assert len(strip) == 10


def test_interpolation_strip_endpoints():
    z1 = torch.randn(256)
    z2 = torch.randn(256)
    strip = interpolation_strip(z1, z2, steps=5, method="slerp")
    assert torch.allclose(strip[0], z1, atol=1e-5)
    assert torch.allclose(strip[-1], z2, atol=1e-5)
