import shutil
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from lightning_fabric.utilities import rank_zero
from omegaconf import DictConfig, OmegaConf

from romil import train_utils


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="training",
)
def main(training_config: DictConfig):
    train_utils.seed_torch(training_config["seed"])

    results_dir = Path(training_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "splits").mkdir(exist_ok=True)

    dataset = hydra.utils.instantiate(
        training_config[training_config["task"]]["MIL_Dataset"]
    )
    folds = np.arange(
        training_config["k_folds"]["k_start"], training_config["k_folds"]["k_end"]
    )
    ckpt_path = training_config.training_args.callbacks.model_checkpoint.dirpath

    mlflow.set_tracking_uri(
        training_config["training_args"]["trainer"]["logger"]["tracking_uri"]
    )
    mlflow.set_experiment(
        training_config["training_args"]["trainer"]["logger"]["experiment_name"]
    )
    rank = rank_zero._get_rank()
    if rank == 0 or rank is None:
        mlflow_run = mlflow.start_run()
        OmegaConf.update(
            training_config,
            "training_args.trainer.logger.run_id",
            mlflow_run.info.run_id,
        )
        with open(results_dir / "mlflow_run_id.txt", "w", encoding="utf-8") as file:
            file.write(mlflow_run.info.run_id)

    for fold in folds:
        split_csv_filename = (
            Path(training_config["split_dir"])
            / f"{training_config['task']}_{training_config['label_frac']}"
            / f"splits_{fold}.csv"
        )
        shutil.copy(
            split_csv_filename,
            results_dir / "splits" / f"splits_{fold}.csv",
        )
        OmegaConf.update(
            training_config,
            "training_args.callbacks.model_checkpoint.dirpath",
            f"{ckpt_path}/fold_{fold}",
        )
        OmegaConf.update(
            training_config, "training_args.trainer.logger.prefix", f"fold_{fold}"
        )
        fold_test_metrics = train_utils.train(
            dataset, fold, training_config, split_csv_filename
        )

        fold_test_metrics["fold"] = fold
        pd.DataFrame([fold_test_metrics]).to_parquet(
            results_dir / "test_metrics.parquet", partition_cols=["fold"]
        )
    mlflow.end_run()


if __name__ == "__main__":
    main()  # pylint: disable=E1120
