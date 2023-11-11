import os
import subprocess
from glob import glob
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from romil import patch2grid


@hydra.main(
    config_path="../conf",
    config_name="preprocessing",
    version_base=None,
)
def pipeline(cfg: DictConfig):
    """performs both patch extraction and feature computation"""

    dir_svs = cfg["patching"]["dir_svs"]
    preprocessing_savedir = cfg["patching"]["preprocessing_savedir"]
    step_size = cfg["patching"]["step_size"]
    patch_size = cfg["patching"]["patch_size"]
    desired_mag = cfg["patching"]["desired_mag"]
    preset = cfg["patching"]["preset"]
    use_simple_seg = cfg["patching"]["use_simple_seg"]

    batch_size = cfg["features"]["batch_size"]
    feat_dir = cfg["features"]["feat_dir"]
    lst_dirs = glob(dir_svs + "/*/")

    lst_df = []
    for dirclass in lst_dirs:
        classname = Path(dirclass).stem
        savedirclass = os.path.join(preprocessing_savedir, classname)
        cmd = (
            f"python CLAM/create_patches_fp.py --source {dirclass} --save_dir"
            f" {savedirclass} --patch --seg --step_size {step_size} --patch_size"
            f" {patch_size} --desired_mag {desired_mag}"
            f" --preset {preset} --use_simple_seg {use_simple_seg}"
        )
        print(cmd)
        cmd_list = cmd.split()
        subprocess.run(cmd_list)

        csv_path = os.path.join(savedirclass, "slides.csv")
        cmd = (
            "python CLAM/extract_features_fp.py --data_h5_dir"
            f" {savedirclass} --data_slide_dir {dirclass} --csv_path"
            f" {csv_path} --feat_dir {feat_dir} --batch_size {batch_size}"
        )
        cmd_list = cmd.split()
        print(cmd)
        subprocess.run(cmd_list)

        df = pd.read_csv(os.path.join(savedirclass, "slides.csv"))
        df["case_id"] = df["slide_id"]
        df["label"] = classname
        lst_df.append(df)

    df_final = pd.concat(lst_df, ignore_index=True)
    df_final.to_csv(os.path.join(feat_dir, "slides.csv"), index=False)

    patch2grid.main(cfg)


if __name__ == "__main__":
    pipeline()
