from typing import Dict, List

import torch
from torch import nn
from xformers.ops import fmha

from romil.models import dsmil
from romil.models.sine_embedding import SinePositionalEmbedding


class RoPEAMIL(nn.Module):
    def __init__(
        self,
        positional_encoder: nn.Module,
        mil_head: nn.Module,
        input_dim: int,
        hidden_dim: int,
        absolute_position_embeddings: bool,
    ):
        """Whole model with position encoder and abmil head

        Args:
            positional_encoder (nn.Module): RoFormerEncoder
            mil_head (nn.Module): ABMIL_MB or ABMIL_SB from models.ABMIL_heads.ppy
            input_dim (int)
            hidden_dim (int)
            absolute_position_embeddings (bool): Apply sin/cos absolute position encoding
        """
        super().__init__()
        self.positional_encoding = positional_encoder
        self.mil_head = mil_head
        self.instance_classifiers = mil_head.instance_classifiers
        self.n_classes = self.mil_head.n_classes
        self.dim_reduction_projection = nn.Linear(input_dim, hidden_dim)

        if absolute_position_embeddings:
            self.absolute_position_embeddings = SinePositionalEmbedding(hidden_dim)
        else:
            self.absolute_position_embeddings = None

    def instance_loss(
        self,
        features: torch.Tensor,
        attention_scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.mil_head.instance_loss(features, attention_scores, labels)

    def forward(
        self, features: List[torch.Tensor], coords: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:

            features (List[torch.Tensor]): (b * (Nb, c))
            coords (List[torch.Tensor]): (b * (Nb, 2)) for patches with non-zero features


        Returns:
            torch.Tensor: logits (b,n_classes)
            attention_scores (b,n_classes, n_patches)
            updated_features (b,n_patches, hidden_dim)
        """
        attn_bias, padded_features = fmha.BlockDiagonalMask.from_tensor_list(
            [feature.unsqueeze(0) for feature in features]
        )  # (1, T, attention_dim) with T = sum(Nb)

        attn_bias, padded_coords = fmha.BlockDiagonalMask.from_tensor_list(
            [coord.unsqueeze(0) for coord in coords]
        )  # (1, T, 2) with T = sum(Nb)

        padded_features = self.dim_reduction_projection(padded_features)

        if self.absolute_position_embeddings is not None:
            padded_features = self.absolute_position_embeddings(
                padded_features, padded_coords
            )

        encoded_features = self.positional_encoding(
            padded_features, padded_coords, attn_bias=attn_bias
        )

        logits, attention_scores, updated_features = self.mil_head(
            encoded_features, attn_bias
        )
        # output to match the clam output in case of instance loss
        return {
            "logits": logits,
            "attention_scores": attention_scores,
            "updated_features": updated_features,
        }


class RoPEDSMIL(nn.Module):
    def __init__(
        self,
        positional_encoder: nn.Module,
        mil_head: dsmil.MILNet,
        input_dim: int,
        hidden_dim: int,
        absolute_position_embeddings: bool,
    ):
        """Whole model with position encoder and dsmil head

        Args:
            positional_encoder (nn.Module): RoFormerEncoder
            mil_head (dsmil.MILNet): DSMIL
            input_dim (int)
            hidden_dim (int)
            absolute_position_embeddings (bool): Apply sin/cos absolute position encoding
        """
        super().__init__()
        self.positional_encoding = positional_encoder
        self.mil_head = mil_head
        self.instance_classifiers = None
        self.n_classes = self.mil_head.n_classes
        self.dim_reduction_projection = nn.Linear(input_dim, hidden_dim)

        if absolute_position_embeddings:
            self.absolute_position_embeddings = SinePositionalEmbedding(hidden_dim)
        else:
            self.absolute_position_embeddings = None

    def forward(
        self, features: List[torch.Tensor], coords: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:

            features (List[torch.Tensor]): (b * (Nb, c))
            coords (List[torch.Tensor]): (b * (Nb, 2)) for patches with non-zero features


        Returns:
            logits (b,n_classes)
            attention_scores (b,n_classes, n_patches)
            updated_features (b,n_patches, hidden_dim)
            patches_predictions (b, n_patches, n_classes)
        """
        attn_bias, padded_features = fmha.BlockDiagonalMask.from_tensor_list(
            [feature.unsqueeze(0) for feature in features]
        )  # (1, T, attention_dim) with T = sum(Nb)

        attn_bias, padded_coords = fmha.BlockDiagonalMask.from_tensor_list(
            [coord.unsqueeze(0) for coord in coords]
        )  # (1, T, 2) with T = sum(Nb)

        padded_features = self.dim_reduction_projection(padded_features)

        if self.absolute_position_embeddings is not None:
            padded_features = self.absolute_position_embeddings(
                padded_features, padded_coords
            )

        encoded_features = self.positional_encoding(
            padded_features, padded_coords, attn_bias=attn_bias
        )

        (
            patches_predictions,
            logits,
            attention_scores,
            bag_representations,
        ) = self.mil_head(encoded_features, attn_bias)
        return {
            "logits": logits,
            "attention_scores": attention_scores,
            "updated_features": encoded_features,
            "patches_predictions": patches_predictions,
        }
