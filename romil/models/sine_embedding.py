# Adapted from https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/sine.py

import math

import torch
from torch import nn


class SinePositionalEmbedding(nn.Module):
    def __init__(self, dim_model: int, theta: int = 10000):
        super().__init__()
        self.dim_model = dim_model
        self.theta = theta

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Apply sine positional encoding.
        Independently for each dimension (x, y) and concat the generated embeddings

        Adaptation from standard implementation to account for sparse coordinates

        Input batch is expected to have been concatenated along the seq_len dimension

        Args:
            features (torch.Tensor): (1, T, dim)
            coords (torch.Tensor): (1, T, 2)

        Returns:
            torch.Tensor: (1, T, dim)
        """
        seq_len = features.shape[1]
        pos_x = (
            coords.to(features)[:, :, 0].unsqueeze(-1).repeat(1, 1, self.dim_model // 2)
        )  # (1, T, dim)
        pos_y = (
            coords.to(features)[:, :, 1].unsqueeze(-1).repeat(1, 1, self.dim_model // 2)
        )  # (1, T, dim)
        dim = (torch.arange(0, self.dim_model // 2).unsqueeze(0).repeat(seq_len, 1)).to(
            coords
        )  # (T, dim)
        div = torch.exp(
            -math.log(self.theta) * (2 * (dim // 2) / (self.dim_model // 2))
        )  # (T, dim)
        pos_x *= div
        pos_y *= div
        # sin on even indices and cos on uneven indices for both x and y
        pos_x[:, :, 0::2] = torch.sin(pos_x[:, :, 0::2])
        pos_x[:, :, 1::2] = torch.cos(pos_x[:, :, 1::2])
        pos_y[:, :, 0::2] = torch.sin(pos_y[:, :, 0::2])
        pos_y[:, :, 1::2] = torch.cos(pos_y[:, :, 1::2])
        return features + torch.concat((pos_x, pos_y), -1)
