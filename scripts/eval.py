import logging
import shutil
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from romil import eval_utils

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf",
    config_name="eval",
)
def main(eval_config: DictConfig) -> None:
    mlflow.set_tracking_uri(eval_config["tracking_uri"])

    results_dir = Path(eval_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "splits").mkdir()
    (results_dir / "checkpoints").mkdir()

    dataset = hydra.utils.instantiate(eval_config[eval_config["task"]]["MIL_Dataset"])
    datasets_id = {"train": 0, "val": 1, "test": 2, "all": -1}

    folds = np.arange(
        eval_config["k_folds"]["k_start"], eval_config["k_folds"]["k_end"]
    )

    for fold in folds:
        if not eval_config["load_model_from_mlflow"]:
            ckpt = Path(eval_config["ckpt_path"]) / f"fold_{fold}" / "best.ckpt"
        else:
            ckpt = mlflow.artifacts.download_artifacts(
                run_id=eval_config["mlflow"]["run_id"],
                artifact_path=f"fold_{fold}/best.ckpt",
            )

        shutil.copy(
            ckpt,
            results_dir / "checkpoints" / f"best_fold{fold}.ckpt",
        )

        if datasets_id[eval_config["split"]] < 0:
            split_dataset = dataset
        else:
            csv_path = (
                Path(eval_config["split_dir"])
                / f"{eval_config['task']}_{eval_config['label_frac']}"
                / f"splits_{fold}.csv"
            )
            shutil.copy(
                csv_path,
                results_dir / "splits" / f"splits_{fold}.csv",
            )
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[eval_config["split"]]]

        patient_preds, results = eval_utils.eval(split_dataset, eval_config, ckpt)

        patient_preds["fold"] = fold
        patient_preds["split"] = eval_config["split"]

        results["fold"] = fold
        results["split"] = eval_config["split"]

        pd.DataFrame([results]).to_parquet(
            results_dir / "eval_metrics.parquet",
            partition_cols=["fold", "split"],
        )
        patient_preds.to_parquet(
            results_dir / "patients_preds.parquet",
            partition_cols=["fold", "split"],
        )
        log.info(f"fold_{fold} results")
        log.info(results)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
