"""Tests for breeding operators."""

import torch
from breeding.breeding import (
    breed_average,
    breed_crossover,
    breed_block_crossover,
    breed_guided,
    breed_style_mix,
    mutate,
    truncate,
    truncate_w,
)


def test_breed_average_shape():
    a = torch.randn(256)
    b = torch.randn(256)
    child = breed_average(a, b, ratio=0.5)
    assert child.shape == (256,)


def test_breed_average_extremes():
    a = torch.randn(256)
    b = torch.randn(256)
    assert torch.allclose(breed_average(a, b, ratio=0.0), a, atol=1e-5)
    assert torch.allclose(breed_average(a, b, ratio=1.0), b, atol=1e-5)


def test_breed_crossover_shape():
    a = torch.randn(256)
    b = torch.randn(256)
    child = breed_crossover(a, b)
    assert child.shape == (256,)


def test_breed_crossover_contains_parent_values():
    a = torch.zeros(256)
    b = torch.ones(256)
    child = breed_crossover(a, b)
    # Every value should be either 0 or 1
    assert ((child == 0) | (child == 1)).all()


def test_breed_block_crossover_shape():
    a = torch.randn(256)
    b = torch.randn(256)
    child = breed_block_crossover(a, b, num_blocks=4)
    assert child.shape == (256,)


def test_breed_guided_shape():
    a = torch.randn(256)
    b = torch.randn(256)
    child = breed_guided(a, b)
    assert child.shape == (256,)


def test_breed_guided_norm_bounded():
    a = torch.randn(256) * 5
    b = torch.randn(256) * 5
    child = breed_guided(a, b, max_norm=2.5)
    assert torch.norm(child) <= 2.5 + 1e-5


def test_mutate_shape():
    z = torch.randn(256)
    mutated = mutate(z, mutation_rate=0.5, mutation_strength=0.3)
    assert mutated.shape == (256,)


def test_mutate_zero_rate():
    z = torch.randn(256)
    mutated = mutate(z, mutation_rate=0.0, mutation_strength=1.0)
    assert torch.allclose(mutated, z)


def test_truncate():
    z = torch.randn(256) * 10
    truncated = truncate(z, max_norm=2.0)
    assert torch.norm(truncated) <= 2.0 + 1e-5


def test_truncate_no_op_for_small():
    z = torch.randn(256) * 0.1
    truncated = truncate(z, max_norm=2.0)
    assert torch.allclose(truncated, z)


# --- W-space operator tests ---


def test_truncate_w_identity():
    """psi=1.0 should return the input unchanged."""
    w = torch.randn(512)
    w_avg = torch.randn(512)
    result = truncate_w(w, w_avg, psi=1.0)
    assert torch.allclose(result, w)


def test_truncate_w_collapse():
    """psi=0.0 should collapse to w_avg."""
    w = torch.randn(512)
    w_avg = torch.randn(512)
    result = truncate_w(w, w_avg, psi=0.0)
    assert torch.allclose(result, w_avg)


def test_truncate_w_midpoint():
    """psi=0.5 should be the midpoint between w_avg and w."""
    w = torch.randn(512)
    w_avg = torch.zeros(512)
    result = truncate_w(w, w_avg, psi=0.5)
    assert torch.allclose(result, w * 0.5)


def test_style_mix_shape():
    a = torch.randn(512)
    b = torch.randn(512)
    result = breed_style_mix(a, b, num_ws=16, crossover_layer=4)
    assert result.shape == (16, 512)


def test_style_mix_coarse_fine():
    """Coarse layers should come from parent A, fine from parent B."""
    a = torch.zeros(512)
    b = torch.ones(512)
    result = breed_style_mix(a, b, num_ws=16, crossover_layer=4)
    # Layers 0-3 should be all zeros (from parent A)
    assert (result[:4] == 0).all()
    # Layers 4-15 should be all ones (from parent B)
    assert (result[4:] == 1).all()


def test_operators_work_with_512_dim():
    """All operators should work with 512-dim W vectors (not just 256-dim Z)."""
    a = torch.randn(512)
    b = torch.randn(512)
    assert breed_average(a, b).shape == (512,)
    assert breed_crossover(a, b).shape == (512,)
    assert breed_block_crossover(a, b).shape == (512,)
    assert breed_guided(a, b).shape == (512,)
    assert mutate(a).shape == (512,)
