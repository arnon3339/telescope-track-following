import numpy as np
import pandas as pd
import os

from modules import (
    mylplotlib, roothit, noise,
    gatecfg, reconstruction as rec, utils
    )
data_dir = "./data/"
if __name__ == "__main__":
    # for i in range(1000):
    #     gatecfg.run_gate(i)
    #     file_path = f"./data/simulation/pmma-seeds/200000telescope_70MeV_noabs_39mmPMMA_swp{i}.root"
    #     proton_hits = proton_only.get_hit_data(file_path, 100000)
    #     noise_hits = noise.get_hit_data(file_path, 100000)
    #     # print(np.array(proton_hits[proton_hits.layerID == 5]["edep"]))
    #     mylplotlib.plot_edep_hist(proton_hits, f"Proton hits energy deposition seed {i}")
    #     mylplotlib.plot_edep_hist(noise_hits, f"Noise hits energy deposition seed {i}")
    #     with open("./logs/sim0.txt", "w+") as fs:
    #         fs.write(f"seed {i}\n")
    # file_path = "./data/simulation/ens/"
    # file_names = os.listdir(f"{file_path}root/")
    # file_names.sort()
    # for f_i, f in enumerate(file_names):
    #     in_path = f"{file_path}root/{f}"
    #     print(in_path)
    #     out_path = f"{file_path}csv/e70MeV/{f[:-5]}.csv"
    #     proton_hits = roothit.get_hit_simdata(in_path, 10000, "proton")
    #     proton_hits.to_csv(out_path, index=None)
    # proton_hits = roothit.get_hit_simdata(os.path.join(file_path, "root", 
    #                                                    "200000telescope_200MeV_noabs_nodegrader.root"),
    #                                       200000, "proton")
    # proton_hits.to_csv(os.path.join(file_path, "csv", "200000telescope_200MeV_noabs_nodegrader.csv"), 
    #                    index=None)
    # sel_proton_hits0 = roothit.get_selsim_hit0(proton_hits)
    # evnt_list = sel_proton_hits0[sel_proton_hits0.layerID == 0]\
    #     ["eventID"].values.astype(np.int64).tolist()
    # sel_proton_hits0["eventID"] = sel_proton_hits0["eventID"].astype(np.int64)
    # sel_proton_hits = proton_hits[proton_hits.eventID.isin(evnt_list)].copy()
    # print(sel_proton_hits)
    # sel_proton_hits.to_csv("./logs/sel_hits.csv")
    # smaxs = np.linspace(0.2, 20, 100)
    # data_eff_tracks = np.zeros((len(evnt_list), len(smaxs)), dtype=np.int32)
    # for i in range(len(evnt_list)):
    #     for j in range(len(smaxs)):
    #         rec_track = rec.run_rec(sel_proton_hits[sel_proton_hits.eventID\
    #             == evnt_list[i]].copy(), smaxs[j])
    #         rec_data_pd = rec.collect_track_data(rec_track)
    #         if 'MyTrackID' in rec_data_pd.columns:
    #             count_tracks = rec_data_pd.groupby(['MyTrackID'])['MyTrackID'].count()
    #             data_eff_tracks[i, j] = 1 if (count_tracks.values)[0] == 6 else 0
    #     print(f"Finished {i} track.")
    # np.savetxt("./output/trackeff-smax-70MeV.csv", data_eff_tracks)
    # rec_tracks = rec.run_rec(proton_hits, 4.8)
    # mylplotlib.plot_rec_tracks(rec_tracks)
    # proton_hits.to_csv("./output/mon_proton_pd.csv")
    # mylplotlib.plot_mon_tracks(proton_hits, file_name)
    # print(proton_hits)
    # rec.cal_eff_in_smax(proton_hits, output=f"{file_name[:-5]}_smax.csv")
    # proton_hits = roothit.get_hit_simdata(file_path, 200000, "proton")
    # monsel_tracks = utils.get_colmon_tracks(roothit.get_selsim_hit0(proton_hits), proton_hits)
    # mylplotlib.plot_cluster_hist_colmon(utils.get_edep_colmon(monsel_tracks), 
                                        # name="sim_acrylic_70MeV", title="Simualtion tracks 70 MeV")
    # mylplotlib.plot_colmon_track(monsel_tracks)
    # noise_hits = roothit.get_hit_simdata(file_path, 100, "secondary")
    # mylplotlib.plot_tcluster_size_hist(proton_hits, title=f"{file_name[:-5]} proton")
    # mylplotlib.plot_tcluster_size_hist(noise_hits, title=f"{file_name[:-5]} noise")
    # mylplotlib.plot_scatter(proton_hits)
    # mylplotlib.plot_edep_hist(proton_hits, f"edep_proton_70MeV")
    # mylplotlib.plot_nhit([proton_hits[proton_hits.layerID == i]["posX"].values\
    #     for i in range(proton_hits.layerID.max() + 1)][1:], title=f"Proton Simulation: {file_name[:-5]}")
    # utils.root2csv(file_path, particles="proton")
    # pd_data = roothit.get_hit_simdata(file_path, particles="proton")
    # utils.roothit2csv(pd_data)
    # dir_path = "./data/simulation/"
    # file_list = os.listdir(dir_path)
    # file_list.sort()
    # for f in file_list:
    #     print(f)
    #     data = roothit.get_hit_simdata(dir_path + f, particles="proton", num_events=500)
    #     print(utils.get_mean_en(data))
    data = pd.read_csv('./data/simulation/csv/10000telescope_70MeV_noabs_01.csv')
    mylplotlib.plot_sim_hit3D(data[data.eventID < 10], name='simhit')
    # data_proton_hits_70 = pd.read_csv(data_dir + "simulation/csv/10000telescope_70MeV_noabs_00.csv")
    # data_proton_hits_200 = pd.read_csv(data_dir + "simulation/csv/10000telescope_200MeV_noabs_0.csv")
    # mcs_list = np.linspace(0, 0.02, 20)
    # smax_list = np.linspace(0, 0.1, 20)
    # eff_rec_dict = {"mcs": [], "smax": [], "eff": []}
    # for smax_i, smax in enumerate(smax_list):
    #     for mcs_i, mcs in enumerate(mcs_list):
    #         all_track = rec.run_rec(data_proton_hits_70[data_proton_hits_70.eventID < 200],
    #                                 mcs, smax)
    #         eff = rec.get_track_eff(rec.collect_track_data(all_track), 
    #                                 data_proton_hits_70[data_proton_hits_70.eventID < 200].copy())
    #         eff_rec_dict["eff"].append(eff)
    #         eff_rec_dict["mcs"].append(mcs)
    #         eff_rec_dict["smax"].append(smax)
    #         print(f"mcs = {mcs}, smax = {smax}, eff = {eff}")
    # pd.DataFrame(eff_rec_dict).to_csv("./output/data/eff_mcs_smax.csv", index=None)

#     all_tracks_70 = rec.run_rec(data_proton_hits_70[data_proton_hits_70.eventID < 400].copy(),
#                                 0.015, 0.030)
#     all_tracks_200 = rec.run_rec(data_proton_hits_200[data_proton_hits_200.eventID < 400].copy(), 
#                                  0.002, 0.004)
#     collected_tracks_70 = rec.collect_track_data(all_tracks_70)
#     collected_tracks_200 = rec.collect_track_data(all_tracks_200)
#     pd.DataFrame(collected_tracks_70).to_csv("./output/data/reconstruction/\
# tracks70MeV_400p.csv", index=None)
#     pd.DataFrame(collected_tracks_200).to_csv("./output/data/reconstruction/\
# tracks200MeV_400p.csv", index=None)
    # mylplotlib.plot_sim_rec_tracks(rec.collect_track_data(all_track))
    # data_proton_hits_70[data_proton_hits_70.eventID < 10].\
    #     to_csv("./data/simulation/pddata/e70norm_10.csv", index=None)
    # data_70MeV_10event = pd.read_csv("./data/simulation/pddata/e70norm_10.csv", index_col=None)
    # data_proton_hits = []
    # seed_hits_dir = "./data/simulation/rootgate/e70MeVnorm/"
    # seed_data_files = os.listdir(seed_hits_dir)
    # seed_data_files.sort()
    # for f_i, f in enumerate(seed_data_files): 
    #     print(f"Starting with {f}")
    #     root_hit = roothit.get_hit_simdata(seed_hits_dir + f, 10000, "proton")
    #     root_hit["eventID"] = root_hit["eventID"].values + f_i*1e3
    #     root_hit["eventID"] = root_hit["eventID"].astype('int64')
    #     data_proton_hits.append(root_hit)
    #     print(f"Finished {f_i} data")
    # pd.concat(data_proton_hits, ignore_index=True).\
    #     to_csv("./data/simulation/pddata/e70MeVnorm.csv",
    #            index=None)

    # alpide_x0 = utils.cal_comp_x0(
    #     [11, 2.7, 8.897],
    #     [89, 2.33, 9.37]
    #     )
    # sig_70 = utils.cal_highland_angle(70, 0.01, alpide_x0)
    # sig_200 = utils.cal_highland_angle(200, 0.01, alpide_x0)
    # mylplotlib.plot_effsim_contour(pd.read_csv(
    #     "./data/simulation/eff_mcs_smax_200.csv"
    # ), sig = sig_200)
    # rec_tracks_70MeV_data = pd.read_csv("output/data/reconstruction/tracks70MeV_400p.csv", 
    #                                     index_col=None)
    # rec_tracks_200MeV_data = pd.read_csv("output/data/reconstruction/tracks200MeV_400p.csv", 
    #                                     index_col=None)
    # mylplotlib.plot_sim_rec_tracks(rec_tracks_70MeV_data)