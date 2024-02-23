import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import uproot

from modules import (reconstruction as rec, mylplotlib, utils)

# root_data_pmma40 = uproot.open("./data/experiment/PMMA4_0NoABS.root")
# root_data_pmma41 = uproot.open("./data/experiment/PMMA4_1NoABS.root")

pmma40_data = utils.collect_roothits(["PMMA4_0NoABS.root"], "./data/experiment/pmma40/")
pmma41_data = utils.collect_roothits(["PMMA4_1NoABS.root"], "./data/experiment/pmma41/")

# mylplotlib.plot_HoverH(pmma41_data[0], mul=False)

# utils.gen_roots2csv("./data/experiment/pmma41/", name="pmma41")
data40_pd = pd.read_csv("./data/experiment/data_pmma40.csv", index_col=None)
data41_pd = pd.read_csv("./data/experiment/data_pmma41.csv", index_col=None)
# mylplotlib.plot_ncluster_layer(data40_pd)

# track_data_40_pd = pd.read_csv("./data/experiment/reconstruction/ALL/pmma40.csv", index_col=None)
# track_data_41_pd = pd.read_csv("./data/experiment/reconstruction/ALL/pmma41.csv", index_col=None)

track_data_mcs_40_pd = pd.read_csv("./data/experiment/reconstruction/ALL/pmma4_mcs_0.csv", index_col=None)
track_data_mcs_41_pd = pd.read_csv("./data/experiment/reconstruction/ALL/pmma4_mcs_1.csv", index_col=None)

track_data_mcs_40_pd_opt = track_data_mcs_40_pd[track_data_mcs_40_pd.mcs == 0.2516281407035176]
track_data_mcs_41_pd_opt = track_data_mcs_41_pd[track_data_mcs_41_pd.mcs == 0.2075175879396985]

print(f"{len(track_data_mcs_40_pd_opt)}, {len(track_data_mcs_41_pd_opt)}")

mylplotlib.plot_rec_tracks(track_data_mcs_41_pd_opt, name="pmma41")

# print(f"pmma40: {track_data_40_pd['layerID'].max()}, pmma41: {track_data_41_pd['layerID'].max()}")
# mylplotlib.plot_rec_effsmax_pmma([track_data_40_pd, track_data_41_pd])
# mylplotlib.plot_rec_effmcs_pmma([track_data_mcs_40_pd, track_data_mcs_41_pd])

#-------------------------- Ready for track reconstruction -------------------------#
# datahit_40pmma = rec.get_chit_csv(data40_pd)
# datahit_41pmma = rec.get_chit_csv(data41_pd)

# for d_i, d in enumerate([datahit_40pmma, datahit_41pmma]):
#     pd_tracks_list = []
#     for mcs_i, mcs in enumerate(np.linspace(0.001, 0.4, 200)):
#         all_tracks_data = rec.run_rec(
#             d,
#             mcs,
#             0.4
#         ) 
#         pd_tracks_data = rec.collect_track_data(all_tracks_data)
#         if all_tracks_data and 'MyTrackID' in pd_tracks_data.columns:
#             pd_tracks_data.insert(0, "mcs", len(pd_tracks_data.index)*[mcs])
#             pd_tracks_list.append(pd_tracks_data)
#         print(f"Finished MCS: {mcs}")
#     if pd_tracks_list:
#         pd.concat(pd_tracks_list, ignore_index=True).\
#             to_csv(f"./data/experiment/reconstruction/ALL/pmma4_mcs_{d_i}.csv", index=False)
