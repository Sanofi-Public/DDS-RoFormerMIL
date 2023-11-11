import logging
from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from CLAM.utils import file_utils

log = logging.getLogger(__name__)


def coords2grid(xy_coordinates: np.ndarray, patch_size: int) -> np.ndarray:
    """Turns (x,y) coordinates into grid coordinates
    in a grid of step patch_size btwn nodes

    Grid origin is set to the patch with smallest x, y
    Args:
        coords (np.ndarray): coordinates of each patch in x,y space (n,2)
        patch_size (int): interval in (x,y) space btwn two grid nodes
    Returns:
        np.ndarray: coordinates of each patches in the grids (n, 2)
    """
    grid_coords = (xy_coordinates / patch_size).astype(int)
    grid_coords_with_origins = grid_coords - grid_coords.min(0)
    return grid_coords_with_origins


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="preprocessing",
)
def main(preprocessing_config: DictConfig) -> None:
    """Load all h5 files in the features folder
    and update the coords to grid

    Store results in the features folder
    """
    h5_folder = Path(preprocessing_config["features"]["feat_dir"]) / "h5_files"
    output_folder = Path(preprocessing_config["features"]["feat_dir"]) / "h5_files_grid"
    output_folder.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(h5_folder.rglob("*"))):
        with h5py.File(file, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            coords = hdf5_file["scaled_coords"][:]

        grid_coords = coords2grid(
            xy_coordinates=coords,
            patch_size=preprocessing_config["patching"]["patch_size"],
        )
        grid_df = pd.DataFrame(grid_coords)

        if grid_df.duplicated().sum() > 0:
            log.info("%i duplicated patches dropped", grid_df.duplicated().sum())
            grid_coords = grid_df.drop_duplicates().values
            features = np.delete(
                features, grid_df[grid_df.duplicated()].index.values, 0
            )

        output_path = output_folder / file.name
        asset_dict = {"features": features, "coords": grid_coords}
        file_utils.save_hdf5(output_path, asset_dict, attr_dict=None, mode="w")


if __name__ == "__main__":
    main()
