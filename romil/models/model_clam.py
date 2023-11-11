""" Models from the Clam repo, modified to handle batch size >1"""

from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class Attn_Net(nn.Module):
    def __init__(
        self, hidden_dim: int, attention_dim: int, dropout: float, n_classes: int
    ):
        """
        Attention Network without Gating (2 fc layers)
        args:
            L: input feature dimension
            D: hidden layer dimension
            dropout: whether to use dropout (p = 0.25)
            n_classes: number of classes
        """

        super(Attn_Net, self).__init__()
        self.n_classes = n_classes
        self.module = [nn.Linear(hidden_dim, attention_dim), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(dropout))

        self.module.append(nn.Linear(attention_dim, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        attention_scores = self.module(features)  # b, n_patches, n_classes
        attention_scores = attention_scores.masked_fill(
            ~padding_mask.unsqueeze(-1).repeat(1, 1, 2), float("-inf")
        )
        transposed_attention_scores = torch.transpose(
            attention_scores, 1, 2
        )  # b,n_class,n_patches

        softmaxed_attention_scores = F.softmax(
            transposed_attention_scores, dim=-1
        )  # softmax over N

        class_representations = torch.matmul(
            softmaxed_attention_scores, features
        )  # b, n_class, embedding_dim
        return class_representations, transposed_attention_scores


class Attn_Net_Gated(nn.Module):
    def __init__(
        self, hidden_dim: int, attention_dim: int, dropout: float, n_classes: int
    ):
        """
        Attention Network with Sigmoid Gating (3 fc layers) from Clam code
        Modified to handle padded sequences for batch size >1
        args:
            L: input feature dimension
            D: hidden layer dimension
            dropout: whether to use dropout (p = 0.25)
            n_classes: number of classes
        """

        super(Attn_Net_Gated, self).__init__()
        self.n_classes = n_classes

        self.attention_a = nn.Sequential(
            *[nn.Linear(hidden_dim, attention_dim), nn.Tanh()]
        )
        self.attention_b = nn.Sequential(
            *[nn.Linear(hidden_dim, attention_dim), nn.Sigmoid()]
        )
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        self.attention_c = nn.Linear(attention_dim, n_classes)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        a = self.attention_a(features)  # b,n,D
        b = self.attention_b(features)  # b,n,D
        A = a.mul(b)  # b,n,D
        attention_scores = self.attention_c(A)  # b, N , n_classes
        attention_scores = attention_scores.masked_fill(
            ~padding_mask.unsqueeze(-1).repeat(1, 1, self.n_classes), float("-inf")
        )
        transposed_attention_scores = torch.transpose(
            attention_scores, 1, 2
        )  # b,n_class,n_patches

        softmaxed_attention_scores = F.softmax(
            transposed_attention_scores, dim=-1
        )  # softmax over N

        class_representations = torch.matmul(
            softmaxed_attention_scores, features
        )  # b, n_class, embedding_dim
        return class_representations, transposed_attention_scores


class CLAM_SB(nn.Module):
    """Clam predictor as in the original paper
    modified to handle batches of >1 examples
    """

    def __init__(
        self,
        attention_net: nn.Module,
        input_dim: int,
        hidden_dim: int,
        dropout: int,
        n_classes: int,
        instance_classifiers: bool,
    ):
        """
        args:
            gate: whether to use gated attention network
            size_arg: config for network size
            dropout: whether to use dropout
            k_sample: number of positive/neg patches to sample for instance-level training
            dropout: whether to use dropout (p = 0.25)
            n_classes: number of classes
            instance_loss_fn: loss function to supervise instance-level training
            subtyping: whether it's a subtyping problem
        """
        super().__init__()

        self.n_classes = n_classes

        self.input_projection = nn.Sequential(
            *[nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        )

        self.attention_net = attention_net
        assert (
            self.attention_net.n_classes == 1
        ), "For CLAM_SB, AttentionNet should have n_classes = 1"

        self.bag_classifiers = nn.Linear(hidden_dim, self.n_classes)

        # Optional so we don't have unused parameters that mess with DDP
        self.instance_classifiers = None
        if instance_classifiers:
            self.instance_classifiers = nn.ModuleList(
                [nn.Linear(hidden_dim, 2) for _ in range(n_classes)]
            )

    def forward(
        self,
        features: List[torch.Tensor],
        _: torch.Tensor,
    ):
        features = torch.nn.utils.rnn.pad_sequence(
            [batch for batch in features], batch_first=True
        )  # (b, n, input_dim)
        padding_mask = features.sum(-1) != 0
        projected_features = self.input_projection(features)  # b, n, embedding_dim
        class_representations, attention_scores = self.attention_net(
            projected_features, padding_mask
        )  # b,N,n_class

        logits = self.bag_classifiers(class_representations).reshape(
            len(features), self.n_classes
        )

        return {
            "logits": logits,
            "attention_scores": attention_scores,
            "updated_features": projected_features,
        }


class CLAM_MB(nn.Module):
    def __init__(
        self,
        attention_net: nn.Module,
        input_dim: int,
        hidden_dim: int,
        dropout: int,
        n_classes: int,
        instance_classifiers: bool,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.input_projection = nn.Sequential(
            *[nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        )
        self.attention_net = attention_net
        # use an indepdent linear layer to predict each class
        self.bag_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_classes)]
        )
        # Optional so we don't have unused parameters that mess with DDP
        self.instance_classifiers = None
        if instance_classifiers:
            self.instance_classifiers = nn.ModuleList(
                [nn.Linear(hidden_dim, 2) for _ in range(n_classes)]
            )

    def forward(
        self,
        features: List[torch.Tensor],
        _: torch.Tensor,
    ):
        features = torch.nn.utils.rnn.pad_sequence(
            [batch for batch in features], batch_first=True
        )  # (b, n, input_dim)
        padding_mask = features.sum(-1) != 0
        projected_features = self.input_projection(features)  # b, n, embedding_dim
        class_representations, attention_scores = self.attention_net(
            projected_features, padding_mask
        )  # b,n_class, hidden_dim

        logits = torch.hstack(
            [
                self.bag_classifiers[classe](class_representations[:, classe])
                for classe in range(self.n_classes)
            ]
        )  # b, n_class
        return {
            "logits": logits,
            "attention_scores": attention_scores,
            "updated_features": projected_features,
        }
