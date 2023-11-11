"""Adapted from https://github.com/binli123/dsmil-wsi/blob/master/dsmil.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import fmha


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, n_classes):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, n_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(
        self, input_size, n_classes, dropout_v=0.0, nonlinear=True, passing_v=False
    ):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=input_size)

        self.output_inference_weights = False

    def forward(self, feats: torch.Tensor, patch_predictions: torch.Tensor):
        """
        Args:
            feats: (1, N, h)
            patch_predictions: (1, N, n_classes)

        Returns:
            logits (1, n_classes)
            attn_weights (1, n_classes, N)
            bag_representations (1, n_classes, h)
        """
        V = self.v(feats)
        Q = self.q(feats)

        # handle multiple classes without for loop
        _, indices_highests = torch.sort(
            patch_predictions, 1, descending=True
        )  # sort class scores along the instance dimension, m_indices in shape (1, N, n_classes)
        m_feats = torch.index_select(
            feats, dim=1, index=indices_highests[0, 0, :]
        )  # select critical instances, m_feats in shape (1, n_classes, h)
        q_max = self.q(
            m_feats
        )  # compute queries of critical instances, q_max in shape (1, n_classes, Q)

        if not Q.is_cuda or self.output_inference_weights:
            # For unit test purposes as xformers doesn't handle cpu
            # Or inference purposes to get the attn_weights
            # FOR INFERENCE, it has to be batchsize=1 to avoid padding and stuff
            attn_weights = q_max @ Q.transpose(
                1, 2
            )  # compute inner product of Q to each entry of q_max, attn_weightsin shape (1,n_classes, N), each column contains unnormalized attention scores
            attn_weights = F.softmax(
                attn_weights / torch.sqrt(torch.tensor(Q.shape[-1]).to(attn_weights)),
                -1,
            )  # normalize attention scores along patches dimension (sum(patches)=1 for each class)
            bag_representations = (
                attn_weights @ V
            )  # compute bag representation, (1, n_classes, h)
        else:
            attn_weights = torch.tensor([])
            bag_representations = fmha.memory_efficient_attention(q_max, Q, V)
        logits = self.fcc(bag_representations).view(1, -1)  # (1, n_classes)
        return logits, attn_weights, bag_representations


class MILNet(nn.Module):
    def __init__(self, patch_classifier, attention_net, n_classes):
        super(MILNet, self).__init__()
        self.patch_classifier = patch_classifier
        self.attention_net = attention_net
        self.instance_classifiers = None
        self.n_classes = n_classes

    def forward(self, features, _):
        feats, patch_predictions = self.patch_classifier(features)
        prediction_bag, attention_weights, bag_representations = self.attention_net(
            feats, patch_predictions
        )

        return patch_predictions, prediction_bag, attention_weights, bag_representations
