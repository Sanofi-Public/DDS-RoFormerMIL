import subprocess
from pathlib import Path

import numpy as np

from romil.patch2grid import coords2grid


def test_coords2grid():
    patch_size = 100

    coords = np.array([[0, 0], [20, 160], [110, 20], [520, 310]])
    expected = np.array([[0, 0], [0, 1], [1, 0], [5, 3]])
    np.testing.assert_equal(coords2grid(coords, patch_size), expected)

    # Translate everything
    coords = 1000 + coords
    np.testing.assert_equal(coords2grid(coords, patch_size), expected)

    # Translate columns differently
    coords[:, 0] = 200 + coords[:, 0]
    coords[:, 1] = 500 + coords[:, 1]
    np.testing.assert_equal(coords2grid(coords, patch_size), expected)


def test_use_simple_seg(tmp_path):
    # Getting the slide
    slide_dir = tmp_path / Path("slide_dir/slide/slide.svs")
    slide_dir.mkdir(parents=True, exist_ok=True)

    # Using simple seg
    cmd = (
        "python preprocessing_pipeline.py"
        f" patching.dir_svs={tmp_path / Path('slide_dir')} patching.preprocessing_savedir={tmp_path / Path('prepro_svdir') } features.feat_dir={tmp_path / Path('features')} patching.use_simpe_seg=True"
        " patching.preset=tests/simple_seg_false.csv"
    )
    cmd_list = cmd.split()
    subprocess.run(cmd_list)

    # Using clam seg
    cmd = (
        "python preprocessing_pipeline.py"
        f" patching.dir_svs={tmp_path / Path('slide_dir')} patching.preprocessing_savedir={tmp_path / Path('prepro_svdir') } features.feat_dir={tmp_path / Path('features')} patching.use_simpe_seg=False"
        " patching.preset=tests/simple_seg_true.csv"
    )
    cmd_list = cmd.split()
    subprocess.run(cmd_list)
