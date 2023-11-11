from typing import Any

import torch
from torch import nn
from xformers.ops import fmha

from romil.models.attention import Block


class IdentityEncoding(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, features: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        """Dummy positional encoding doing nothing
        features: torch.tensor of any shape
        """
        return features


class RoFormerEncoder(nn.Module):
    """RoFormer
    blocks are instantiated with hydra by default but this can be updated manually
    """

    def __init__(
        self,
        n_attention_block: int,
        n_embd: int,
        n_head: int,
        dropout: float = 0.25,
        bias: bool = True,
        rope: bool = True,
        rope_freqs: str = "pixel",
        resid_dropout: float = 0.25,
    ):
        """Roformer as a list of transformer blocks
        Args:
            n_attention_block (int): number of blocks
            n_embd (int): input/output embedding dim
            n_head (int): number of head in multihead self attention
            dropout (float): dropout in the final MLP proj
            bias (bool): Apply bias in the attention projections
            rope (bool): Apply RoPE or not
            rope_freqs (str): "pixel" or "lang
            resid_dropout (float): dropout on the output of self-attention
        """
        super().__init__()
        self.attention_blocks = nn.ModuleList(
            Block(
                n_embd=n_embd,
                n_head=n_head,
                dropout=dropout,
                bias=bias,
                rope=rope,
                rope_freqs=rope_freqs,
                resid_dropout=resid_dropout,
            )
            for _ in range(n_attention_block)
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask,
    ) -> torch.Tensor:
        for block in self.attention_blocks:
            features = block(
                features=features,
                coords=coords,
                attn_bias=attn_bias,
            )

        return features
