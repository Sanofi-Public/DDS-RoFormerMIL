from pathlib import Path
from typing import Dict, Tuple

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from CLAM.utils import utils
from romil import lightning_utils
from romil.lightning_datamodule import collate_fn
from romil.models.lightning_modules import MILLitModule


def initiate_model(eval_config: DictConfig, ckpt_path: Path) -> MILLitModule:
    """Instantiate model and load checkpoint

    Args:
        eval_config (DictConfig): model config as in model_dict.yaml
        ckpt_path (Path)

    Returns:
        pl.LightningModule: MILLitModule
    """
    model = hydra.utils.instantiate(eval_config["lightning_module"])

    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    return model


def eval(
    dataset: Dataset, eval_config: DictConfig, ckpt_path: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Instantiate the model in eval_config["lightning_module"]
    and run predictions on the dataset

    Args:
        dataset (Dataset)
        eval_config (DictConfig): with ["lightning_module"], ["task"]]["n_classes"], ["multiclass_avg"]
        ckpt_path (Path):

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]:
            df with ["slide_id", "probs", "labels"]
            dict with metrics
    """
    model = initiate_model(eval_config, ckpt_path)

    loader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
    )

    patient_predictions, results = evaluate(model, loader, eval_config)
    return patient_predictions, results


def evaluate(model: pl.LightningModule, loader: DataLoader, eval_config: DictConfig):
    trainer = pl.Trainer(accelerator="gpu", logger=False)

    outputs = trainer.predict(
        model=model,
        dataloaders=loader,
    )
    preds = torch.concat([output["preds"] for output in outputs])
    labels = torch.concat([output["targets"] for output in outputs])
    logits = torch.concat([output["logits"] for output in outputs])
    probs = F.softmax(logits, dim=1)

    loader.dataset.slide_data["probs"] = pd.Series(probs.tolist())
    loader.dataset.slide_data["labels"] = pd.Series(labels.tolist())

    if eval_config[eval_config["task"]]["n_classes"] > 2:
        classif_results = lightning_utils.get_multiclass_classif_results(
            probs,
            labels,
            eval_config[eval_config["task"]]["n_classes"],
            eval_config["multiclass_avg"],
        )
    else:
        classif_results = lightning_utils.get_binary_classif_results(
            probs,
            labels,
        )

    classif_results["error"] = utils.calculate_error(preds, labels)

    return loader.dataset.slide_data[["slide_id", "probs", "labels"]], classif_results
