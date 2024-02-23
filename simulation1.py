import numpy as np
import pandas as pd

from modules import mylplotlib
from modules import (proton_only, noise)
from modules import gatecfg

if __name__ == "__main__":
    # for i in range(1000, 2000):
    #     gatecfg.run_gate(i)
    #     file_path = f"./data/simulation/pmma-seeds/200000telescope_70MeV_noabs_39mmPMMA_swp{i}.root"
    #     proton_hits = proton_only.get_hit_data(file_path, 100000)
    #     noise_hits = noise.get_hit_data(file_path, 100000)
    #     # print(np.array(proton_hits[proton_hits.layerID == 5]["edep"]))
    #     mylplotlib.plot_edep_hist(proton_hits, f"Proton hits energy deposition seed {i}")
    #     mylplotlib.plot_edep_hist(noise_hits, f"Noise hits energy deposition seed {i}")
    #     with open("./logs/sim1.txt", "w+") as fs:
    #         fs.write(f"seed {i}\n")
    for i in range(20):
        gatecfg.run_gate(i)