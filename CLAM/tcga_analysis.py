# %%
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader
from tqdm import tqdm

# %%
dir_tcga = "/home/tcga_nsclc/"

dirs_classes = glob(dir_tcga + "/*/")
dict_paths = {}
lst_mag = []
for indication_path in dirs_classes:
    indication = Path(indication_path).stem
    lst_svs = glob(indication_path + "*.svs")
    dict_paths[indication] = lst_svs

    for psvs in tqdm(lst_svs):
        img = OpenSlideWSIReader(psvs)
        info = img.info.as_dict()
        mag = info["objective_power"]
        if mag is None:
            print("*", info)
        else:
            lst_mag.append(mag)
plt.hist(lst_mag)

# %%
