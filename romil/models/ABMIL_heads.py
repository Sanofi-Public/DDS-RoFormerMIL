import torch
from torch import nn
from xformers.ops import fmha

from romil.models.attention import ClassAttention


class ABMIL_SB(nn.Module):
    def __init__(
        self,
        attention_net: ClassAttention,
        input_dim: int,
        hidden_dim: int,
        dropout: int,
        n_classes: int,
        instance_classifiers: bool,
    ):
        """
        args:
            attention_net: ClassAttention for unpadded xformers attention
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
        features: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask,
    ):
        projected_features = self.input_projection(features)  # 1, T, embedding_dim
        class_representations, attention_scores = self.attention_net(
            projected_features, attn_bias
        )  # b, 1, hidden_dim

        logits = self.bag_classifiers(class_representations).view(
            -1, self.n_classes
        )  # b, n_classes

        return logits, attention_scores, projected_features


class ABMIL_MB(nn.Module):
    """ABMIL head with class attention  with xformer mem-efficient attention"""

    def __init__(
        self,
        attention_net: ClassAttention,
        input_dim: int,
        hidden_dim: int,
        dropout: int,
        n_classes: int,
        instance_classifiers: bool,
    ):
        """
        args:
            attention_net: ClassAttention for unpadded xformers attention
        """
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
        features: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask,
    ):
        """
        Instances of elements in the batch are concatenated into a single sequence
        Class attention applied using the block diagonal mask to have one prediction
        for each sample

        Args:
            features (torch.Tensor): (1, T, emb_dim)
                            concatenated features of all samples in the batch
            attn_bias (fmha.BlockDiagonalMask): block diagonal mask to separate samples

        Returns:
            _type_: _description_
        """
        projected_features = self.input_projection(features)  # 1, T, embedding_dim
        class_representations, attention_scores = self.attention_net(
            projected_features, attn_bias
        )  # b,n_class, hidden_dim

        logits = torch.hstack(
            [
                self.bag_classifiers[classe](class_representations[:, classe])
                for classe in range(self.n_classes)
            ]
        )  # b, n_class
        return logits, attention_scores, projected_features
