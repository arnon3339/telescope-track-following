#!/home2/arnon3339/Projects/telescope-track-following/.python/bin/python3
import sys
import pandas as pd
import numpy as np
from modules import reconstruction as rec

def run_rec_with_noisecut():
    noisecut = -1
    try:
        if len(sys.argv) > 1:
            noisecut = int(sys.argv[1])
    except:
        pass
    for_track_data = pd.read_csv("./newdata/datasubselchit_70MeV1000MU.csv", index_col=None)
    for_track_data["posX"] = for_track_data["cposX"]*30.0/1024
    for_track_data["posY"] = for_track_data["cposY"]*13.8/512
    for_track_data["posZ"] = for_track_data["layerID"]*25.0 + 50
    if noisecut < 0:
        for_track_data = pd.concat([
            for_track_data[(for_track_data.layerID != 4) & (for_track_data.clusterSize >= 2)],
            for_track_data[(for_track_data.layerID == 4) & (for_track_data.clusterSize >= 3)]
            ], ignore_index=True)

    else:
        print(noisecut)
        for_track_data = for_track_data[for_track_data.clusterSize >= noisecut]
    evts = np.unique(for_track_data["eventID"].values)
    rec_track_list = [] 
    evts.sort()
    smaxs = np.linspace(0, 15, 100)
    mcss = np.linspace(0, 3.14*80/180, 100)
    for smax in smaxs:
        for mcs in mcss:
            print(f"Starting Smax: {smax}, MCS: {mcs}")
            for evt_i, evt in enumerate(evts):
                for_track_data_evt = for_track_data[for_track_data.eventID == evt]
                rec_data = rec.run_rec(for_track_data_evt, mcs, smax)
                collected_track_data = rec.collect_track_data(rec_data)
                if not collected_track_data.empty:
                    collected_track_data.insert(0, "mcs", [mcs]*len(collected_track_data.index))
                    collected_track_data.insert(0, "smax", [smax]*len(collected_track_data.index))
                else:
                    continue
                # print(collected_track_data)
                rec_track_list.append(collected_track_data)
            print(f"Finished Smax: {smax}, MCS: {mcs}\n")
    track_result_data = pd.concat(rec_track_list, ignore_index=True)
    del track_result_data["Unnamed: 0"]
    track_result_data["mcs"] = track_result_data["mcs"].apply(lambda x: '{:.4f}'.format(x))
    track_result_data["smax"] = track_result_data["smax"].apply(lambda x: '{:.4f}'.format(x))
    track_result_data.to_csv(f"./newdata/reconstruction/sub/e70MeV_{'normal' if noisecut==-1 else noisecut}.csv",
               index=False)

if __name__ == "__main__":
    run_rec_with_noisecut()