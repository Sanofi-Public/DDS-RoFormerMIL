import logging
import os
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from CLAM.datasets.dataset_generic import Generic_MIL_Dataset
from romil.lightning_datamodule import MILDatamodule

log = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    dataset: Generic_MIL_Dataset, fold: int, args: DictConfig, split_csv_filename: Path
) -> Dict[str, float]:
    """
    train for a single fold
    """
    print("\nTraining Fold {}!".format(fold))

    if (
        args["training_args"]["lightning_module"]["model"]["_target_"]
        == "models.model_bpa.RPEDSMIL"
    ):
        log.info(
            "DSMIL currently doesn't support batch size >1. Setting"
            " accumulate_grad_batches instead"
        )
        OmegaConf.update(
            args,
            "training_args.trainer.accumulate_grad_batches",
            args["training_args"]["datamodule_params"]["batch_size"],
        )
        OmegaConf.update(args, "training_args.datamodule_params.batch_size", 1)

    datamodule = MILDatamodule(
        dataset, split_csv_filename, **args["training_args"]["datamodule_params"]
    )

    model = hydra.utils.instantiate(args["training_args"]["lightning_module"])

    callbacks = [
        hydra.utils.instantiate(callback_cfg)
        for _, callback_cfg in args["training_args"]["callbacks"].items()
    ]

    trainer = hydra.utils.instantiate(
        args["training_args"]["trainer"], callbacks=callbacks
    )
    if fold == 0:
        trainer.logger.log_hyperparams(args["training_args"])
    trainer.logger.experiment.log_artifact(
        trainer.logger.run_id, split_csv_filename, f"fold_{fold}"
    )

    trainer.fit(model=model, datamodule=datamodule)

    return trainer.test(ckpt_path="best", datamodule=datamodule)[0]


def seed_torch(seed=7):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
