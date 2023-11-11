from math import ceil
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="create_splits",
)
def main(splits_config: DictConfig) -> None:
    slides_csv_path = splits_config["csv_path"]
    all_slides_df = pd.read_csv(slides_csv_path)
    shuffle = splits_config["shuffle"]
    label_fracs = splits_config["label_frac"]
    if shuffle:
        random_state = splits_config["seed"]
    else:
        random_state = None

    if splits_config["stratify"]:  # stratify between train+val and test, wrt label
        kf = StratifiedKFold(
            n_splits=splits_config["k_folds"],
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        kf = KFold(
            n_splits=splits_config["k_folds"],
            shuffle=shuffle,
            random_state=random_state,
        )

    for fold, (train_index, test_index) in enumerate(
        kf.split(X=all_slides_df["slide_id"], y=all_slides_df["label"])
    ):
        for label_frac in label_fracs:
            train_df = all_slides_df.iloc[train_index]["slide_id"].values
            test_df = all_slides_df.iloc[test_index]["slide_id"].values

            train_np, val_np = train_test_split(
                train_df,
                test_size=splits_config["val_frac"],
                random_state=splits_config["seed"],
            )
            del train_df  # to not confuse it with train_np

            training_size = ceil(label_frac * train_np.shape[0])
            train_np = train_np[
                :training_size
            ]  # taking the first training_size elements to actually train on

            this_fold_df = pd.DataFrame(
                {
                    "train": pd.Series(train_np),
                    "val": pd.Series(val_np),
                    "test": pd.Series(test_df),
                }
            )
            split_dir = (
                Path(splits_config["split_dir"])
                / f"{splits_config['task']}_{int(100*label_frac)}"
            )
            split_dir.mkdir(parents=True, exist_ok=True)
            this_fold_df.to_csv(split_dir / f"splits_{fold}.csv")
            this_fold_descriptor = generate_descriptor_one_fold(
                this_fold_df, all_slides_df
            )
            this_fold_descriptor.to_csv(split_dir / f"splits_{fold}_descriptor.csv")


def generate_descriptor_one_fold(
    fold_df: pd.DataFrame, all_slides_df: pd.DataFrame
) -> pd.DataFrame:
    unique_labels = all_slides_df["label"].unique()
    out_df = pd.DataFrame({"label": [], "train": [], "val": [], "test": []})
    for label in unique_labels:
        # for each label (row in the final df) we count how many slides with said label are in train, val, and test
        n_train = 0
        n_val = 0
        n_test = 0

        for slide in fold_df[
            "train"
        ].dropna():  # counting how many of the train slides have the label
            this_slide_label = all_slides_df.loc[
                all_slides_df["slide_id"] == slide, "label"
            ].values[0]
            if this_slide_label == label:
                n_train += 1

        for slide in fold_df[
            "val"
        ].dropna():  # counting how many of the val slides have the label
            this_slide_label = all_slides_df.loc[
                all_slides_df["slide_id"] == slide, "label"
            ].values[0]
            if this_slide_label == label:
                n_val += 1

        for slide in fold_df[
            "test"
        ].dropna():  # counting how many of the test slides have the label
            this_slide_label = all_slides_df.loc[
                all_slides_df["slide_id"] == slide, "label"
            ].values[0]
            if this_slide_label == label:
                n_test += 1

        out_df.loc[len(out_df.index)] = [label, n_train, n_val, n_test]
    out_df.set_index("label")
    return out_df


if __name__ == "__main__":
    results = main()  # pylint: disable=E1120
