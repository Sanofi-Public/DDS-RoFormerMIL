from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from datasets.dataset_generic import save_splits


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="create_splits",
)
def main(splits_config: DictConfig) -> None:
    dataset = hydra.utils.instantiate(
        splits_config[splits_config["task"]]["MIL_Dataset"]
    )

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.round(num_slides_cls * splits_config["val_frac"]).astype(int)
    test_num = np.round(num_slides_cls * splits_config["test_frac"]).astype(int)

    for frac in splits_config["label_frac"]:
        split_dir = (
            Path(splits_config["split_dir"])
            / f"{splits_config['task']}_{int(frac * 100)}"
        )
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset.create_splits(
            k=splits_config["k_folds"],
            val_num=val_num,
            test_num=test_num,
            label_frac=frac,
        )
        for i in range(splits_config["k_folds"]):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(
                splits,
                ["train", "val", "test"],
                split_dir / f"splits_{i}.csv",
            )
            save_splits(
                splits,
                ["train", "val", "test"],
                split_dir / f"splits_{i}_bool.csv",
                boolean_style=True,
            )
            descriptor_df.to_csv(split_dir / f"splits_{i}_descriptor.csv")


if __name__ == "__main__":
    results = main()  # pylint: disable=E1120
