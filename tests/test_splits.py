import subprocess
from glob import glob
from pathlib import Path
from random import randint

import pandas as pd


def test_all_data_in_test(tmp_path):
    """
    testing that all slides are present exactly once in the test set across all splits
    """
    create_dummy_slides_csv(tmp_path)
    results = tmp_path
    csv_path = results / Path("slides.csv")

    cmd = (
        "python scripts/new_create_splits_seq.py"
        f" split_dir={results} results_dir={results} csv_path={csv_path} task=test"
    )
    cmd_list = cmd.split()
    subprocess.run(cmd_list)

    task = Path(
        "test_100"
    )  # only testing the case where label_frac = 100 because the others are subsets of it
    splits_path = Path(results / task)
    test_slides_df = pd.DataFrame({"slide_id": []})

    for split in glob(str(splits_path) + "/splits_?.csv"):
        df_i = pd.read_csv(split)
        df_i = pd.DataFrame(df_i["test"].dropna().rename("slide_id"))
        test_slides_df = pd.concat([test_slides_df, df_i])
    assert not test_slides_df.empty
    all_slides_df = pd.read_csv(csv_path).astype("float64")
    # Sorting both dfs because we want to know if they contain the same info, regardless of order

    test_slides_df = test_slides_df.sort_values(by="slide_id").reset_index(drop=True)
    all_slides_df = all_slides_df.sort_values(by="slide_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(test_slides_df, all_slides_df["slide_id"].to_frame())


def test_dataset_size(tmp_path):
    """
    verifying that size(initial dataset) = size(train) + size(validation) + size(test) for each fold
    """
    create_dummy_slides_csv(tmp_path)
    results = tmp_path
    csv_path = results / Path("slides.csv")

    cmd = (
        "python new_create_splits_seq.py"
        f" split_dir={results} results_dir={results} csv_path={csv_path} task=test"
    )
    cmd_list = cmd.split()
    subprocess.run(cmd_list)

    slides_path = results / Path("slides.csv")
    task = Path("test_100")
    splits_path = Path(results / task)

    all_slides_df = pd.read_csv(slides_path)
    total_size = len(all_slides_df)

    for split in glob(str(splits_path) + "/splits_?.csv"):
        df_i = pd.read_csv(split)

        size_train = len(df_i["train"].dropna())
        size_val = len(df_i["val"].dropna())
        size_test = len(df_i["test"].dropna())

        assert total_size == size_train + size_val + size_test


def test_duplicates(tmp_path):
    """Make sure no slide is present twice in the dataset each time (dataset = train + val + test)"""
    create_dummy_slides_csv(tmp_path)
    results = tmp_path
    csv_path = results / Path("slides.csv")

    cmd = (
        "python new_create_splits_seq.py"
        f" split_dir={results} results_dir={results} csv_path={csv_path} task=test"
    )
    cmd_list = cmd.split()
    subprocess.run(cmd_list)

    task = Path("test_100")
    splits_path = Path(results / task)
    for split in glob(str(splits_path) + "/splits_?.csv"):
        df_i = pd.read_csv(split)
        all_the_slides = pd.Series([])
        all_the_slides = pd.concat(
            [all_the_slides, df_i["train"], df_i["val"], df_i["test"]]
        )
        assert (not all_the_slides.duplicated().unique()[0]) and (
            len(all_the_slides.duplicated().unique() == 1)
        )
        # duplicated gives a series that says for each value whether or not it is duplicated
        # then we want it to contain only false values so we get the first value of unique and make sure it's the only one


def create_dummy_slides_csv(save_dir: Path):
    """
    Creating a dummy file called slides.csv and using it instead of our own file
    Makes a csv with 1000 dummy values that belong to two classes: 0 and 1
    """
    slides_df = pd.DataFrame({"slide_id": [], "case_id": [], "label": []})
    for i in range(1000):
        slides_df.loc[len(slides_df.index)] = [i, i, randint(0, 1)]
    slides_df.to_csv(save_dir / "slides.csv")
