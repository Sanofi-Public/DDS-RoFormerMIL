# Adapted from https://github.com/karpathy/nanoGPT/


import math
from typing import Tuple

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F
from xformers.ops import fmha

from romil.models import rotary_embedding


def inefficient_scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    inefficient sdpa implementation to allow to get attn weigths at inference for viz

    Same inputs than https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    Args:
        q : (N,...,L,E)
        k : (N,...,S,E).
        v : (N,...,S,E).
        attn_mask : (N,...,L,S) float tensor with 0 where attention should be applied and -inf for masked tokens
    return:
        att_output: (N,...,L,E)
        att_weigths: (N,...,L,S)
    """
    attn_weight = torch.softmax(
        (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1
    )
    return attn_weight @ v, attn_weight


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, n_embd, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        rope: bool,
        rope_freqs: str,
    ):
        """Self-attention class leveraging xformers mem-efficient attention
        On CPU, attention is performed with torch 2.0 sdpa (not with efficient kernel)

        Inspired from Nano-GPT

        Args:
            n_embd (int): embedding dimension
            n_head (int): number of heads to splits the embeddings
            bias (bool): Wether to use bias in projections
            rope (bool): Wether to use rotary position encoding
            rope_freqs (str): which freqs to use for rope
                value should be either
                    - lang for the OG rotary (as in the paper, in xformers and the lang of lucidrains)
                    - pixel for the freqs defined in https://github.com/lucidrains/rotary-embedding-torch/
        """
        super().__init__()
        assert n_embd % n_head == 0
        self.input_projection = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        self.output_projection = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

        self.n_head = n_head
        self.n_embd = n_embd

        if rope:
            self.rope = rotary_embedding.RotaryEmbedding(
                dim_model=self.n_embd // self.n_head, freqs_for=rope_freqs
            )
        else:
            self.rope = None

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask,
    ) -> torch.Tensor:
        """Self attention with rotary positional encoding

        Args:
            features (torch.Tensor): (1, T, n_embd)
            coords (torch.Tensor): (1, T, 2)
            rotary_pos_emb_freqs (torch.Tensor): (n,d/2)
            attn_bias (fmha.BlockDiagonalMask)
        Returns:
            torch.Tensor: (1, T, n_embd)
        """

        # calculate query, key, values projections
        q, k, v = self.input_projection(features).split(self.n_embd, dim=-1)

        k = k.view(
            1, features.shape[1], self.n_head, self.n_embd // self.n_head
        )  # (1, T, nh, hs)
        q = q.view(
            1, features.shape[1], self.n_head, self.n_embd // self.n_head
        )  # (1, T, nh, hs)

        if self.rope is not None:
            q = self.rope(q.transpose(1, 2), coords).transpose(1, 2)
            k = self.rope(k.transpose(1, 2), coords).transpose(1, 2)

        v = v.view(1, features.shape[1], self.n_head, self.n_embd // self.n_head)

        if not q.is_cuda:
            ### For unit test purposes
            ### as xformers doesn't handle cpu
            out = (
                torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_mask=attn_bias.materialize((q.shape[1], k.shape[1])),
                    dropout_p=0,
                )
                .transpose(1, 2)
                .reshape(1, -1, self.n_embd)
            )
        else:
            out = fmha.memory_efficient_attention(q, k, v, attn_bias=attn_bias).view(
                1, features.shape[1], self.n_embd
            )
        return self.resid_dropout(self.output_projection(out))


class ClassAttention(nn.Module):
    def __init__(
        self,
        n_classes: int,
        hidden_dim: int,
        attention_dim: int,
        n_head: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        assert attention_dim % n_head == 0

        self.keys_projection = nn.Linear(hidden_dim, attention_dim, bias=bias)
        self.values_projection = nn.Linear(hidden_dim, attention_dim, bias=bias)
        self.output_projection = nn.Linear(attention_dim, hidden_dim, bias=bias)
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.resid_dropout = nn.Dropout(dropout)

        self.n_head = n_head
        self.dropout = dropout
        self.n_classes = n_classes

        self.class_tokens = nn.Parameter(
            nn.init.xavier_normal_(
                torch.rand((1, n_classes, attention_dim), requires_grad=True)
            )
        )

        self.output_inference_weights = False

    def forward(
        self, features: torch.Tensor, attn_bias: fmha.BlockDiagonalMask
    ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): (1, T, d)
            attn_bias BlockDiagonalMask

        Returns:
            torch.Tensor: (b,n_class,d)
        """
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        keys = self.keys_projection(features)  # (1, T, attention_dim)
        values = self.values_projection(features)  # (1, T, attention_dim)

        k = keys.view(
            1, features.shape[1], self.n_head, self.attention_dim // self.n_head
        )  # (1, T, nh, hs)
        q = (
            self.class_tokens.view(
                1, self.n_classes, self.n_head, self.attention_dim // self.n_head
            )
            .repeat(1, len(attn_bias._batch_sizes), 1, 1)
            .to(k)
        )  # (1, n_classes * b, nh, hs)
        v = values.view(
            1, features.shape[1], self.n_head, self.attention_dim // self.n_head
        )  # (1, T, nh, hs)

        class_attn_bias = fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=[self.n_classes] * len(attn_bias._batch_sizes),
            kv_seqlen=[el[1] - el[0] for el in attn_bias.k_seqinfo.intervals()],
        )

        attn_weights = torch.zeros([])
        if not q.is_cuda or self.output_inference_weights:
            # For unit test purposes as xformers doesn't handle cpu
            # Or inference purposes to get the attn_weights
            # FOR INFERENCE, it has to be batchsize=1 to avoid padding and stuff
            out, attn_weights = inefficient_scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=class_attn_bias.materialize((q.shape[1], k.shape[1])).to(q),
            )
            out = out.transpose(1, 2).reshape(
                len(attn_bias._batch_sizes), self.n_classes, self.attention_dim
            )
            # Attention weights for each token are averaged over the attention heads
            attn_weights = einops.rearrange(attn_weights, "b h c n -> b c n h")
        else:
            out = fmha.memory_efficient_attention(
                q, k, v, attn_bias=class_attn_bias
            ).view(len(attn_bias._batch_sizes), self.n_classes, self.attention_dim)
        return self.output_projection(out), attn_weights


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        rope: bool,
        rope_freqs: str,
        resid_dropout: float = 0,
    ):
        """Transformer block

        Inspired from Nano-GPT

        Args:
            n_embd (int): embedding dimension
            n_head (int): number of heads to splits the embeddings
            bias (bool): Wether to use bias in projections
            rope (bool): Wether to use rotary position encoding
            rope_freqs (str): which freqs to use for rope
                value should be either
                    - lang for the OG rotary (as in the paper, in xformers and the lang of lucidrains)
                    - pixel for the freqs defined in https://github.com/lucidrains/rotary-embedding-torch/
            resid_dropout (fload): value for self-attention residual dropout
        """
        super().__init__()
        self.ln_1 = LayerNorm(n_embd=n_embd, bias=bias)
        self.attn = SelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            dropout=resid_dropout,
            bias=bias,
            rope=rope,
            rope_freqs=rope_freqs,
        )  # mem efficient doesn't support dropout
        self.ln_2 = LayerNorm(n_embd=n_embd, bias=bias)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask,
    ) -> torch.Tensor:
        attention_output = self.attn(
            features=self.ln_1(features),
            coords=coords,
            attn_bias=attn_bias,
        )
        features = features + attention_output
        features = features + self.mlp(self.ln_2(features))
        return features
