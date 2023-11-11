import math

import numpy as np
import pytest
import torch

from romil.models.sine_embedding import SinePositionalEmbedding


def test_run() -> None:
    """test that embeddings can be generated for arbitrary dims"""
    dim = 32
    steps = 400
    n_patches = 100
    batch_size = 1  # Xformers batching concat all elements into (1, T, ndim) tensors

    features = torch.randn((batch_size, n_patches, dim))
    coords = torch.randint(0, steps, (batch_size, n_patches, 2)).float()

    embedding = SinePositionalEmbedding(dim)

    output = embedding(features, coords)

    assert output.shape == features.shape


def test_equals() -> None:
    """test that similar inputs yields similar outputs"""
    dim = 32
    steps = 400
    n_patches = 100
    batch_size = 8  # Xformers batching concat all elements into (1, T, ndim) tensors

    features = torch.randn((n_patches, dim)).repeat(batch_size, 1).unsqueeze(0)
    coords = (
        torch.randint(0, steps, (n_patches, 2))
        .float()
        .repeat(batch_size, 1)
        .unsqueeze(0)
    )

    embedding = SinePositionalEmbedding(dim)

    output = embedding(features, coords)

    for i in range(1, batch_size):
        assert np.array_equal(
            output[:, i * n_patches : (i + 1) * n_patches].cpu().numpy(),
            output[:, :n_patches].cpu().numpy(),
        )


@pytest.mark.parametrize("theta", (10000, 5000, 10))
def test_values(theta) -> None:
    """test that added bias values correspond to the formula"""
    dim = 32
    steps = 400
    n_patches = 100
    features = torch.zeros((1, n_patches, dim))
    coords = torch.randint(0, steps, (1, n_patches, 2)).float()

    embedding = SinePositionalEmbedding(dim, theta=theta)
    output = embedding(features, coords)

    for patch in range(n_patches):
        for i in range(0, dim // 2, 2):
            assert (
                abs(
                    output[0, patch, i]
                    - math.sin(
                        coords[0, patch, 0]
                        * math.exp(-math.log(theta) * (2 * (i // 2) / (dim // 2)))
                    ),
                )
                < 1e-4
            )

            assert (
                abs(
                    output[0, patch, i + 1]
                    - math.cos(
                        coords[0, patch, 0]
                        * math.exp(-math.log(theta) * (2 * (i // 2) / (dim // 2)))
                    ),
                )
                < 1e-4
            )

            assert (
                abs(
                    output[0, patch, (dim // 2) + i]
                    - math.sin(
                        coords[0, patch, 1]
                        * math.exp(-math.log(theta) * (2 * (i // 2) / (dim // 2)))
                    ),
                )
                < 1e-4
            )

            assert (
                abs(
                    output[0, patch, (dim // 2) + i + 1]
                    - math.cos(
                        coords[0, patch, 1]
                        * math.exp(-math.log(theta) * (2 * (i // 2) / (dim // 2)))
                    ),
                )
                < 1e-4
            )
