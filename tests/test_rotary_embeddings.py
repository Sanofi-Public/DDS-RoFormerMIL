import pytest
import torch

from romil.models.rotary_embedding import RotaryEmbedding

dtype = torch.float16
torch.manual_seed(0)
torch.set_default_device("cpu")


@pytest.mark.parametrize("n_patches", (10, 50, 100))
@pytest.mark.parametrize("translation", (-100, -50, 0, 5, 10))
@pytest.mark.parametrize("freqs_for", ("pixel", "lang"))
def test_translation_invariance(
    n_patches: int, translation: int, freqs_for: str
) -> None:
    """test that rotary embeddings yields same dot products attentions when translating the coordinates"""
    dim = 32
    steps = 400

    features = torch.randn((1, n_patches, dim))
    coords = torch.randint(
        max(0, -translation), min(steps, steps - translation), (1, n_patches, 2)
    )
    translated_coords = coords + translation

    pos_emb = RotaryEmbedding(dim_model=dim, freqs_for=freqs_for)

    q = pos_emb(features, coords)
    k = pos_emb(features, coords)
    dots = torch.einsum("b n d, b m d -> b n m", q, k)

    translated_q = pos_emb(features, translated_coords)
    translated_k = pos_emb(features, translated_coords)
    translated_dots = torch.einsum("b n d, b m d -> b n m", translated_q, translated_k)
    assert torch.allclose(translated_dots, dots, atol=7e-4)

    new_coords = torch.randint(0, 2 * steps, (1, n_patches, 2))
    q = pos_emb(features, new_coords)
    k = pos_emb(features, new_coords)
    new_dots = torch.einsum("b n d, b m d -> b n m", q, k)
    assert not (torch.allclose(new_dots, dots))


@pytest.mark.parametrize("freqs_for", ("pixel", "lang"))
def test_new_max_length(freqs_for):
    ### Rotary has cached sin and cos.
    ### Test that passing longer sequences works
    dim = 32
    steps = 40
    n_patches = 30

    pos_emb = RotaryEmbedding(dim_model=dim, freqs_for=freqs_for)

    features = torch.randn((1, n_patches, dim))
    coords = torch.randint(0, steps, (1, n_patches, 2))
    pos_emb(features, coords)

    # Try again with same max(coords)
    pos_emb(features, coords)

    # Try again with max(coords)= 1 + max(rotary coords)
    coords = coords + 1
    pos_emb(features, coords)

    # Try again with max(coords) >> max(rotary coords)
    coords = coords * 2
    pos_emb(features, coords)
