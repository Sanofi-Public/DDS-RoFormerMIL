from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def instance_loss(
    features: torch.Tensor,
    attention_scores: torch.Tensor,
    label: torch.Tensor,
    instance_classifiers: nn.ModuleList,
    n_classes: int,
    subtyping: bool,
    k_sample: int,
    instance_loss_fn: Callable,
) -> torch.Tensor:
    """
    https://github.com/mahmoodlab/CLAM/
    Args:
        attention_scores (torch.Tensor): (b, n_classes, n)
        features (torch.Tensor): (b, n, d)
        label (torch.Tensor): (b) arg from the clam forward pass

    Returns:
        torch.Tensor: instance loss
    """
    ### Copied from CLAM model
    ### Simply iterate over the batches to get the batch instance loss as original implementation doensn't support it

    inst_labels = F.one_hot(
        label, num_classes=n_classes
    )  # binarize label (b, n_classes)

    A = F.softmax(attention_scores, dim=1)  # (b, n_classes, n)
    inst_losses = []
    for i, classifier in enumerate(instance_classifiers):
        for batch in range(len(features)):
            inst_label = inst_labels[batch, i].item()
            if inst_label == 1:  # in-the-class:
                instance_loss, preds, targets = inst_eval(
                    A[batch], features[batch], classifier, k_sample, instance_loss_fn
                )
            else:  # out-of-the-class
                if subtyping:
                    instance_loss, preds, targets = inst_eval_out(
                        A[batch],
                        features[batch],
                        classifier,
                        k_sample,
                        instance_loss_fn,
                    )
                else:
                    continue
        inst_losses += [instance_loss]
    total_inst_loss = torch.stack(inst_losses, dim=0).sum(dim=0)
    if subtyping:
        total_inst_loss /= n_classes
    return total_inst_loss


def create_positive_targets(length: int) -> torch.Tensor:
    return torch.full((length,), 1).long()


def create_negative_targets(length: int) -> torch.Tensor:
    return torch.full((length,), 0).long()


# instance-level evaluation for in-the-class attention branch
def inst_eval(
    attention_scores: torch.Tensor,
    features: torch.Tensor,
    classifier: nn.ModuleList,
    k_sample: int,
    instance_loss_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """https://github.com/mahmoodlab/CLAM/

    Args:
        attention_scores (torch.Tensor): (b, n, n_classes)
        features (torch.Tensor): (b, n, d)
        classifier (nn.ModuleList): list of classifiers for each classes
        k_sample (int): int
        instance_loss_fn (Callable): Loss
    """
    if len(attention_scores.shape) == 1:
        attention_scores = attention_scores.view(1, -1)
    top_p_ids = torch.topk(attention_scores, k_sample)[1][-1]
    top_p = torch.index_select(features, dim=0, index=top_p_ids)
    top_n_ids = torch.topk(-attention_scores, k_sample, dim=1)[1][-1]
    top_n = torch.index_select(features, dim=0, index=top_n_ids)
    p_targets = create_positive_targets(k_sample)
    n_targets = create_negative_targets(k_sample)

    all_targets = torch.cat([p_targets, n_targets], dim=0)
    all_instances = torch.cat([top_p, top_n], dim=0)
    logits = classifier(all_instances)
    all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
    instance_loss = instance_loss_fn(logits, all_targets.to(logits.device))
    return instance_loss, all_preds, all_targets


# instance-level evaluation for out-of-the-class attention branch
def inst_eval_out(
    attention_scores: torch.Tensor,
    features: torch.Tensor,
    classifier: nn.ModuleList,
    k_sample: int,
    instance_loss_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """https://github.com/mahmoodlab/CLAM/

    Args:
        attention_scores (torch.Tensor): (b, n, n_classes)
        features (torch.Tensor): (b, n, d)
        classifier (nn.ModuleList): list of classifiers for each classes
        k_sample (int): int
        instance_loss_fn (Callable): Loss
    """
    if len(attention_scores.shape) == 1:
        attention_scores = attention_scores.view(1, -1)
    top_p_ids = torch.topk(attention_scores, k_sample)[1][-1]
    top_p = torch.index_select(features, dim=0, index=top_p_ids)
    p_targets = create_negative_targets(k_sample)
    logits = classifier(top_p)
    p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
    instance_loss = instance_loss_fn(logits, p_targets.to(logits.device))
    return instance_loss, p_preds, p_targets
