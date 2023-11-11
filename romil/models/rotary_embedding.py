# Inspired by https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
# and https://github.com/lucidrains/rotary-embedding-torch

from math import pi

import torch
from einops import rearrange


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(1, T, dim) -> (1, T, dim)"""
    x = rearrange(x, "... (d r) -> ... d r", r=2)  # (1, T, dim) -> (1, T, dim/2, 2)
    x1, x2 = x.unbind(dim=-1)  # 2 * (1, T, dim/2)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")  # (1, T, dim)


def apply_rotary_pos_emb(
    features: torch.Tensor, coords: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    features (torch.Tensor): (..., T, dim)
    coords (torch.Tensor): (..., T, 2)
    cos/sin (torch.Tensor): (maxT, dim/2)

    Returns:
            torch.Tensor: (..., T, dim)
    """
    cos = cos[coords].flatten(-2)  # (1, T, 2, dim/2)->(1, T, dim)
    sin = sin[coords].flatten(-2)  # (1, T, 2, dim/2)->(1, T, dim)
    return (features * cos) + (rotate_half(features) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int, freqs_for="lang", max_freq=400):
        """2D rope embeddings based on x, y coordinates (!= causal indices in nlp tokens)

        dim_model/2 embedded based on x coords
        dim_model/2 embedded based on y coords

        Args:
            dim_model (int): _description_
            freqs_for (str):
                lang is Theta from sin/cos absolute.
                pixel is from https://github.com/lucidrains/rotary-embedding-torchs
            max_freq (int): number of coords to start cache with
        """
        super().__init__()
        self.max_freq = max_freq
        self.freqs_for = freqs_for

        # Generate and save the inverse frequency buffer (non trainable)
        if freqs_for == "lang":
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, dim_model // 2, 2).float() / (dim_model // 2))
            )
        elif freqs_for == "pixel":
            inv_freq = torch.linspace(1.0, max_freq / 2, dim_model // 4) * pi

        self.register_buffer("inv_freq", inv_freq)
        self.max_coords = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, coords):
        seq_len = coords.max().item()
        if seq_len > self.max_coords:
            self.max_coords = seq_len
            t = torch.arange(self.max_coords + 1).to(
                coords
            )  # Arange(max) returns [0,..., max-1], we need [0,..., max]
            if self.freqs_for == "pixel":
                t = t / self.max_freq
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.repeat_interleave(freqs, 2, -1)
            self._cos_cached = emb.cos()  # (seq_len, n_dim/2)
            self._sin_cached = emb.sin()  # (seq_len, n_dim/2)

        return self._cos_cached, self._sin_cached

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """

        Args:
            features (torch.Tensor): (..., T, dim)
            coords (torch.Tensor): (... T, 2)

        Returns:
            torch.Tensor: position encoded features (..., T, dim)
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(coords)

        return apply_rotary_pos_emb(
            features, coords, self._cos_cached, self._sin_cached
        )
