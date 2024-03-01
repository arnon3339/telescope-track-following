#!/home2/arnon3339/Projects/telescope-track-following/.python/bin/python3
from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from itertools import combinations
from modules import (cluster, mylplotlib, roothit, utils, noise, projections as proj,
                     analyze, reconstruction as rec, bg, physics as phys)
import os
import random
import json
import warnings
import uproot
from itertools import combinations

warnings.filterwarnings("ignore")

with open("./config.json", 'r') as f:
    cfg = json.load(f)

if __name__ == "__main__":
    lim_dict = {
        "70 MeV": {"a":40, "b": 20,"x":500, "y": 267},
        "200 MeV": {"a":50, "b": 25,"x":510, "y": 262}
    }
#------------- Optimising BG -----------------#
#     file_path = "/home/arnon/Projects/telescope-track-\
# following/data/experiment/BG/run426192011_221022192017/"
#     data = utils.get_bg_data(file_path)
#     data_bg = []
#     for i in data:
#         data_bg.append(utils.get_dead_pixel(i))
#     with open("./output/bg_pixel.csv", "w") as fs:
#         writer = csv.writer(fs)
#         writer.writerows(data_bg)
    
    # mylplotlib.plot_bg_hist(data)

#-------------- Evaluating clusters -----------------#

#     file_name = "2255.root"
#     file_path = f"/home/arnon/Projects/telescope-track-\
# following/data/experiment/hits/stack/Col200MeV500MU/{file_name}"
#     clusters_data = cluster.get_clusters(file_path, lim=lim_dict["70 MeV"], cut_size=2)
#     bg.remove_bg(clusters_data)
#     chit_data = rec.get_chit_data([cluster.get_HfromC(d) for d in clusters_data], clusters_data)
#     rec_data = rec.run_rec(chit_data, 4)
#     rec_data_pd = rec.collect_track_data(rec_data)
#     rec_pixdata_pd = rec_data_pd.copy()
#     rec_pixdata_pd["posX"] = rec_pixdata_pd["posX"] * 1024/30
#     rec_pixdata_pd["posY"] = rec_pixdata_pd["posY"] * 512/13.8
#     mylplotlib.plot_rec_tracks(rec_data, scatter=True, kind="200 MeV Experiment pixel", inpixel=True)
#     mylplotlib.plot_nclusters(clusters_data[1:], title=f"Experiment: {file_name[:-5]}")

#--------------------- Multiple events trackt --------------------#
    # file_name = "1395.root"
#     file_path = "/home/arnon/Projects/telescope-track-\
# following/data/experiment/hits/stack/Col070MeV1000MU/"
#     file_list = os.listdir(file_path)
#     file_list.sort()
#     cluster_tracks = []
#     for i in range(len(file_list)):
#         clusters_data = cluster.get_clusters(file_path + file_list[i], lim=lim_dict["70 MeV"], cut_size=2)
#         if clusters_data and clusters_data[0]:
#             bg.remove_bg(clusters_data)
#             chit_data = rec.get_chit_data([cluster.get_HfromC(d) for d in clusters_data], clusters_data)
#             rec_data = rec.run_rec(chit_data, 4.5)
#             rec_data_pd = rec.collect_track_data(rec_data)
#             if "cSize" in rec_data_pd.keys():
#                 cluster_tracks += rec_data_pd["cSize"].values.tolist()
#             print(f"{file_list[i]} is reconstructed.")
#     mylplotlib.plot_csize_hist(cluster_tracks, name="experiment_csize_track_200MeV", 
#                                title="Experiment 200 MeV")

#---------------- Find complete tracks -------------------#
#     file_path = "/home/arnon/Projects/telescope-track-\
# following/data/experiment/hits/stack/Col200MeV500MU/"
#     file_list = os.listdir(file_path)
#     file_list.sort()
#     hit_tracks = []
#     # c_tracks = []
#     for i in range(len(file_list)):
#         clusters_data = cluster.get_clusters(file_path + file_list[i], lim=lim_dict["70 MeV"], cut_size=2)
#         tracks = []
#         # cs = []
#         if clusters_data and clusters_data[0]:
#             bg.remove_bg(clusters_data)
#             chit_data = rec.get_chit_data([cluster.get_HfromC(d) for d in clusters_data], clusters_data)
#             hit_tracks.append([int(file_list[i][:-5]), len(chit_data[chit_data.layerID == 0].values)])
#             for smax in np.linspace(0.2, 10, 50):
#                 rec_data = rec.run_rec(chit_data, smax)
                # rec_data_pd = rec.collect_track_data(rec_data)
#                 if 'MyTrackID' in rec_data_pd.columns:
#                     count_tracks = rec_data_pd.groupby(['MyTrackID'])['MyTrackID'].count()
#                     tracks.append(count_tracks.values.tolist())
#                     # cs.append(rec_data_pd.groupby(['MyTrackID'])['MyTrackID'].values.tolist())
#                 else:
#                     tracks.append([0])
#                     # cs.append([0])
#             print(f"{file_list[i]} is reconstructed.")
#         if tracks:
#             tarr_col = max([len(tt) for tt in tracks])
#             tarr_row = len(tracks)
#             tarr_comp = np.zeros((tarr_row, tarr_col))
#             for r_t in range(len(tracks)):
#                 tarr_comp[r_t][:len(tracks[r_t])] = tracks[r_t]
            # carr_col = max([len(cc) for cc in cs])
            # carr_row = len(cs)
            # carr_comp = np.ones((carr_row, carr_col))*(-1)
            # for r_c in range(len(cs)):
            #     carr_comp[r_c][:len(cs[r_c])] = cs[r_c]
    #         np.savetxt(f"./output/e200MeV/evt{file_list[i][:-5]}.csv", tarr_comp)
    # np.savetxt(f"./output/e200MeVlayer0.csv", np.array(hit_tracks))

#------------------ Compare number of clusters ---------------------#
#     clusters = []
#     file_names = [
#         "PMMA4_0NoABS.root",
#         "PMMA4_1NoABS.root",
#         "PMMA4_2NoABS.root",
#         "Cu70MeV.root"
#     ]
#     file_path = "/home/arnon/Projects/telescope-track-\
# following/data/experiment/"
#     for i in range(len(file_names)):
#         fpath = file_path + file_names[i]
#         clusters.append(cluster.get_clusters(fpath, cut_size=0))
    # mylplotlib.plot_cmp_clusters(clusters)
    # mylplotlib.plot_cluster_4hist(clusters)

    # mylplotlib.plot_cluster_hist(clusters[-1])

#----------------- Plot stack histogram --------------#
#     stack_path = "/home/arnon/Projects/telescope-track-\
# following-old/data/experiment/hits/stack/"
#     stack_dirs = os.listdir(stack_path)
#     stack_dirs.sort()
#     stack_data = []
#     for i in range(len(stack_dirs)):
#         stack_files = os.listdir(stack_path + stack_dirs[i])
#         stack_files.sort()
#         stack_data.append(utils.collect_roothits(stack_files, 
#                                            stack_path + stack_dirs[i] + r"/"))
#     x_prms = analyze.get_gfit(stack_data[0][1][:, 0])
#     y_prms = analyze.get_gfit(stack_data[0][1][:, 1])
#     print(f"{x_prms[1]} {y_prms[1]}")
    # mylplotlib.plot_HoverH(np.array(stack_data[0][1]), hover=False,
    #                        name="Multiple70MeVCu",
    #                        title="Treatment beam in 10 frames")
#     data_path = "/home/arnon/Projects/telescope-track-\
# following/data/experiment/col200MeV/"
#     file_names = os.listdir(data_path)
#     for f_i, f in enumerate(file_names):
#         data = utils.collect_roothits([f"{f}"], 
#                                         "/home/arnon/Projects/telescope-track-\
# following/data/experiment/col200MeV/")
        # print(len(data))
    # for j in range(len(data)):
        # if len(data):
        #     if len(data[0]):
        #         mylplotlib.plot_HoverH(np.array(data[0]), hover=False, mul=False,
        #                                 name="Single70MeVCu")


#---------------------------  Plot beam sigma --------------------------#
    # file_path = "./data/experiment/"
    # beam_data = {
    #     "70 MeV": pd.read_csv(file_path + "beam_70MeV.csv"),
    #     "200 MeV": pd.read_csv(file_path + "beam_200MeV.csv")
    #     }
    # data_70MeV = beam_data["70 MeV"].iloc[:, 3:].values
    # data_200MeV = beam_data["200 MeV"].iloc[:, 3:].values
    # mylplotlib.plot_beam_sigma(data_70MeV, 
    #                            name="beam_col_sigma_70MeV", title="70 MeV") 

#------------------ check complete tracks -------------------#
    # np_dict = utils.count_cmpl_tracks("./output/e200MeV/")
    # eff_dict = utils.get_smax_tracks(np_dict, nhit=7)
    # mylplotlib.plot_contour_cmpl(eff_dict, name="200MeV", title="200MeV", nhit=7)
    # mylplotlib.plot_eff_smax(eff_dict, name="200MeV", title="200MeV")
    # mylplotlib.plot_nchit("./output/e200MeVlayer0.csv")
    # print(count)
    # cmpl = utils.get_smax_tracks("./output/e200MeV_diag/", "./output/e200MeVlayer0_diag.csv")
    # mylplotlib.plot_contour_cmpl(cmpl)

#----------------- Plot removed BG stack histogram --------------#
# for stack_file in ["Col070MeV1000MU", "Col100MeV500MU", "Col120MeV500MU", 
#                    "Col150MeV500MU", "Col180MeV500MU", "Col200MeV500MU"]:
#     stack_path = f"/home/arnon/Projects/telescope-track-following-old\
# /data/experiment/hits/stack/{stack_file}_merged/"
#     utils.gen_roots2csv(stack_path, name=f"{stack_file}")

#------------------ Merge data ------------------------__#
    # utils.merge_evnt_data([
    #     "/home/arnon/Projects/telescope-track-following-old/data/experiment/hits/stack/Col200MeV500MU",
    #     "/home/arnon/Projects/telescope-track-following-old/data/experiment/hits/stack/Col200MeV500MU_2"
    # ])

# ----------------- Evaluating beam sigma ----------------------# 
    # data = {
    #     70:  pd.read_csv("./data/experiment/data_Col070MeV1000MU.csv", index_col=0),
    #     100: pd.read_csv("./data/experiment/data_Col100MeV500MU.csv", index_col=0),
    #     120: pd.read_csv("./data/experiment/data_Col120MeV500MU.csv", index_col=0),
    #     150: pd.read_csv("./data/experiment/data_Col150MeV500MU.csv", index_col=0),
    #     180: pd.read_csv("./data/experiment/data_Col180MeV500MU.csv", index_col=0),
    #     200: pd.read_csv("./data/experiment/data_Col200MeV500MU.csv", index_col=0)
    # }

    # data_sel_1sigma = {
    #     70: utils.get_hits_sigma(data[70], cfg["experiment"]["70 MeV"], area=1), 
    #     100 : utils.get_hits_sigma(data[100], cfg["experiment"]["100 MeV"], area=1), 
    #     120 : utils.get_hits_sigma(data[120], cfg["experiment"]["120 MeV"], area=1), 
    #     150 : utils.get_hits_sigma(data[150], cfg["experiment"]["150 MeV"], area=1), 
    #     180 : utils.get_hits_sigma(data[180], cfg["experiment"]["180 MeV"], area=1), 
    #     200 : utils.get_hits_sigma(data[200], cfg["experiment"]["200 MeV"], area=1), 
    # }

    # data_sel_2sigma = {
    #     70 : utils.get_hits_sigma(data[70], cfg["experiment"]["70 MeV"]),
    #     100 : utils.get_hits_sigma(data[100], cfg["experiment"]["100 MeV"]),
    #     120 : utils.get_hits_sigma(data[120], cfg["experiment"]["120 MeV"]),
    #     150 : utils.get_hits_sigma(data[150], cfg["experiment"]["150 MeV"]),
    #     180 : utils.get_hits_sigma(data[180], cfg["experiment"]["180 MeV"]),
    #     200 : utils.get_hits_sigma(data[200], cfg["experiment"]["200 MeV"]),
    # }

    # data_sel_1sigma = utils.get_concE_data(data_sel_1sigma)
    # data_sel_2sigma = pd.read_csv("./data/experiment/data_sel_2sigma.csv") 
    # data_sel_2sigma = utils.get_concE_data(data_sel_2sigma)
    # data_sel_2sigma.to_csv("./data/experiment/data_sel_2sigma.csv", index=False)
    # roothit.get_hitexp_data(cluster.get_cluster_only(data_sel_2sigma)).\
    #     to_csv("./data/experiment/data_sel_2sigma_hits.csv")
    # rec.gen_rec_mulexp_mulsmax(pd.read_csv("./data/experiment/data_sel_2sigma_hits.csv"), 
    #                            energies=[70, 200])
    # rec_data = utils.get_rected_data(energies=[70, 200])
    # rec_eff_data = analyze.get_mulevt_tract_eff(rec_data)
    # mylplotlib.plot_rec_effsmax(rec_eff_data, name="eff_smax_lines", maxx=300)
    # rec_tracks_dir = r"./data/experiment/reconstruction/ALL/"
    # pd_70_data = [pd.read_csv(f"{rec_tracks_dir}e70/{f}") for f in os.listdir(f"{rec_tracks_dir}e70")]
    # pd_200_data = [pd.read_csv(f"{rec_tracks_dir}e200/{f}") for f in os.listdir(f"{rec_tracks_dir}e200")]
    # pd_data_conc = {70: pd.concat(pd_70_data, ignore_index=True),
    #                 200: pd.concat(pd_200_data, ignore_index=True)}
    # msc_angle = {70: np.unique(pd_data_conc[70]["MSCangle"].values),
    #              200: np.unique(pd_data_conc[200]["MSCangle"].values)}
    # for msc_k, msc_v in msc_angle.items():
    #     evts = np.unique(pd_data_conc[msc_k]["eventID"].values)[:10]
    #     print(evts)
    #     for msc in msc_v:
    #         data_plot = pd_data_conc[msc_k][(pd_data_conc[msc_k].MSCangle == msc) &\
    #             (pd_data_conc[msc_k].eventID.isin(evts))]
    #         mylplotlib.plot_rec_tracks(data_plot, inpixel=True, name=f"e{msc_k}_msc{msc}")
    # mylplotlib.plot_rec_effsmax(rec_eff_data[200], name="e200MeV_smax", title="200 MeV")
    # nhits0_data_70 = pd.read_csv("./data/experiment/reconstruction/e70_nhits0.csv", index_col=None)
    # nhits0_data_200 = pd.read_csv("./data/experiment/reconstruction/e200_nhits0.csv", index_col=None)
    # print(nhits0_data.iloc[:, :].values)
    # mylplotlib.plot_nhits0(
    #     [nhits0_data_70.iloc[:, :].values.astype(np.int64),
    #      nhits0_data_200.iloc[:, :].values.astype(np.int64),
    #      ],
    #     name="hist0_70_200", title="")
    # utils.get_forbid_hits(data_sel_2sigma, energies=[70, 200]).\
    #     to_csv("./data/experiment/reconstruction/forbidden_hits.csv", index=False)
    # forbid_datat = pd.read_csv("./data/experiment/reconstruction/forbidden_hits.csv")
    # forbid_data = forbid_datat
    # count = 0
    # evts_set_df = []
    # for e in [70, 200]:
    #     evts_set = set(())
    #     for evt_i, evt in enumerate(np.unique(forbid_data["eventID"].values)):
    #         if np.unique(forbid_data[forbid_data.eventID == evt]["layerID"].values).size == 1:
    #             # print(f"{np.unique(forbid_data[forbid_data.eventID == evt]['layerID'].values)}: {evt}")
    #             evts_set.add(evt)
    #             count += 1
    #         elif 0 not in np.unique(forbid_data[forbid_data.eventID == evt]["layerID"].values):
    #             evts_set.add(evt)
    #             # print(np.unique(forbid_data[forbid_data.eventID == evt]["layerID"].values))
    #         elif 1 not in np.unique(forbid_data[forbid_data.eventID == evt]["layerID"].values):
    #             evts_set.add(evt)
    #             # print(np.unique(forbid_data[forbid_data.eventID == evt]["layerID"].values))
    #     evts_set_df.append(forbid_data[(forbid_data.energy == e) & (forbid_data.eventID.isin(list(evts_set)))])
    # pd.concat(evts_set_df, ignore_index=True).to_csv("./data/experiment/reconstruction/fail_hits.csv")
    # print(len(evts_set))
    # fail_datat = pd.read_csv("./data/experiment/data_sel_2sigma_hits.csv")
    # fail_data = fail_datat[(fail_datat.eventID.isin([evt for evt in evts_set]))]
    # fail_data.to_csv("./data/experiment/reconstruction/fail_hits.csv", index=False)
    # print([np.unique(h) for h in fail_data[fail_data.layerID == 0]["ID"].values if len(np.unique(h)) > 1])
    # data_sel_2sigma.to_csv("./data/experiment/data_sel_2sigma.csv")
    # data_sel_2sigma.to_csv("./data/experiment/data_sel_2sigma_200.csv")
    # mylplotlib.plot_cluster_count(data_sel_2sigma)
    # mylplotlib.plot_exp_nclusters(data_sel_2sigma)
    # mylplotlib.plot_2D_beamsigma(data_sel_2sigma)
    # data_sel_2sigma = pd.read_csv("./data/experiment/data_sel_2sigma_hits.csv")
    # mylplotlib.plotavg_edep_and_cluster(data_sel_2sigma, cfg["simulation"], name="2sigma", title="")
    # print(sel_70_2sig)
    # for e in np.unique(data_sel_1sigma["energy"].values):
    #     sp = '  '
    #     if e != 200:
    #         mylplotlib.plot_hist_cluster(data_sel_1sigma[data_sel_1sigma.energy == e], name=f"1sigma_{e}MeV", 
    #                                      title=f"{e}MeV", sp=sp)
    # for e in np.unique(data_sel_2sigma["energy"].values):
    #     sp = '  '
    #     if e != 200:
    #         mylplotlib.plot_hist_cluster(data_sel_2sigma[data_sel_2sigma.energy == e], name=f"2sigma_{e}MeV", 
    #                                      title=f"{e}MeV", sp=sp)
    #     else: 
    #         mylplotlib.plot_hist_cluster(data_sel_2sigma[data_sel_2sigma.energy == e], name=f"2sigma_{e}MeV", 
    #                                      title=f"{e}MeV")
    # mylplotlib.plotbox_edep_and_cluster(data_sel_1sigma, name=f"1sigma", title=f"1$\sigma$")
    # mylplotlib.plotbox_edep_and_cluster(data_sel_2sigma, name=f"2sigma", title=f"2$\sigma$")
    # data_fit = {
    #     70: pd.DataFrame(analyze.get_gfit(data[70], lim=cfg["experiment"]["70 MeV"]["siglim"],
    #                                       expected=cfg["experiment"]["70 MeV"]["expected"])),
    #     100: pd.DataFrame(analyze.get_gfit(data[100], lim=cfg["experiment"]["100 MeV"]["siglim"],
    #                                       expected=cfg["experiment"]["100 MeV"]["expected"])),
    #     120: pd.DataFrame(analyze.get_gfit(data[120], lim=cfg["experiment"]["120 MeV"]["siglim"],
    #                                       expected=cfg["experiment"]["120 MeV"]["expected"], bi=True)),
    #     150: pd.DataFrame(analyze.get_gfit(data[150], lim=cfg["experiment"]["150 MeV"]["siglim"],
    #                                       expected=cfg["experiment"]["150 MeV"]["expected"], bi=True)),
    #     180: pd.DataFrame(analyze.get_gfit(data[180], lim=cfg["experiment"]["180 MeV"]["siglim"],
    #                                       expected=cfg["experiment"]["180 MeV"]["expected"], bi=True)),

    # }
    # data_fit_sub = {
    #     120: pd.DataFrame(analyze.fit_sub(data[120], lims=[cfg["experiment"]["120 MeV"]["siglim"], 
    #                                                        cfg["experiment"]["70 MeV"]["siglim"]],
    #                                       expected=cfg["experiment"]["120 MeV"]["expected"])),
    #     150: pd.DataFrame(analyze.fit_sub(data[150], lims=[cfg["experiment"]["150 MeV"]["siglim"], 
    #                                                        cfg["experiment"]["70 MeV"]["siglim"]],
    #                                       expected=cfg["experiment"]["120 MeV"]["expected"])),
    #     180: pd.DataFrame(analyze.fit_sub(data[180], lims=[cfg["experiment"]["180 MeV"]["siglim"], 
    #                                                        cfg["experiment"]["70 MeV"]["siglim"]],
    #                                       expected=cfg["experiment"]["120 MeV"]["expected"])),
    #     200: pd.DataFrame(analyze.fit_sub(data[200], lims=[cfg["experiment"]["200 MeV"]["siglim"], 
    #                                                        cfg["experiment"]["70 MeV"]["siglim"]],
    #                                       expected=cfg["experiment"]["200 MeV"]["expected"])),

    # }
    # for i in [120, 150, 180, 200]:
    #     mylplotlib.plot_beam_dist(data[i], data_fit_sub[i], lim_fits=cfg["experiment"]["70 MeV"]["siglim"],
                                #   name=f"{i}_single", pdf=False, cmin=True)
    # data_fit70 = pd.DataFrame(analyze.get_gfit(data[70], lim=cfg["experiment"]["70 MeV"]["siglim"],
    #                                       expected=cfg["experiment"]["70 MeV"]["expected"]))


    # data_fit_200_single = pd.DataFrame(analyze.fit_sub(data[200], lims=[cfg["experiment"]["200 MeV"]["siglim"], 
    #                                                       cfg["experiment"]["70 MeV"]["siglim"]],
    #                                       expected=cfg["experiment"]["200 MeV"]["expected"]))

    # mylplotlib.plot_beam_dist(data[70], data_fit[70], lim_fits=cfg["experiment"]["70 MeV"]["siglim"],
    #                             name=f"70", pdf=False, cmin=False)
    # mylplotlib.plot_beam_dist(data[120], data_fit[120], lim_fits=cfg["experiment"]["120 MeV"]["siglim"],
    #                             name=f"120", pdf=False, cmin=False, bi=True)
    # mylplotlib.plot_beam_dist(data[180], data_fit[180], lim_fits=cfg["experiment"]["180 MeV"]["siglim"],
    #                             name=f"180", pdf=False, cmin=False, bi=True)
    # mylplotlib.plot_beam_dist(data[200], data_fit_sub[200], lim_fits=cfg["experiment"]["70 MeV"]["siglim"],
    #                             name=f"200_single", pdf=False, cmin=True)
    # mylplotlib.plot_beam_dist(data[200], data_fit[200], lim_fits=cfg["experiment"]["200 MeV"]["siglim"], 
    #                           name=f"200_cim", pdf=False, cmin=True, bi=True)
    # utils.gen_2sigma_data(data, pd.read_csv("./data/experiment/col_beam_data.csv"))
    # mylplotlib.plot_2sigma_entries(data, pd.read_csv("./data/experiment/col_beam_data.csv"))

    #mylplotlib.plot_beam_sigma(cfg["experiment"], name="sigma_beam_fit_pct", pct_fit=True)

    # sigma_data_dict = {"energy": [], "sigmaX": [], "sigmaY": [],
    #                    "meanX": [], "meanY": [], "layerID": []}
    
    # for i in [70, 100, 120, 150, 180, 200]:
    #     for j in range(6):
    #         sigma_data_dict["layerID"].append(j)
    #         sigma_data_dict["energy"].append(i)
    #         sigma_data_dict["sigmaX"].append(data_fit[i][data_fit[i].layerID == j]["paramsX"].values[0][1]) 
    #         sigma_data_dict["sigmaY"].append(data_fit[i][data_fit[i].layerID == j]["paramsY"].values[0][1]) 
    #         sigma_data_dict["meanX"].append(data_fit[i][data_fit[i].layerID == j]["paramsX"].values[0][0]) 
    #         sigma_data_dict["meanY"].append(data_fit[i][data_fit[i].layerID == j]["paramsY"].values[0][0]) 
    #     mylplotlib.plot_beam_dist(data[i], data_fit[i], lim_fits=lim_fits[i],
    #                             name=f"{i}_t_cim_f", pdf=False, cmin=True)
    #     print(i)
    # pd.DataFrame(sigma_data_dict).to_csv("./data/experiment/col_beam_data.csv")
    # mylplotlib.plot_beam_sigma([data_fit70, data_fit_200_single],
    #                            name="beam_sigma_70_200")
    # mylplotlib.plot_beam_dist(data[150], data_fit[150], lim_fits=lim_fits[70],
    #                             name="150_t", pdf=False, cmin=False, bi=True)
    # pd.DataFrame(analyze.get_gfith(data[70], lim_fit)).\
    #     to_csv("./data/experiment/analyse/col_beam_70MeV_h.csv") 
    # pd.DataFrame(analyze.get_gfith(data[70], lim_fit)).\
    #     to_csv("./data/experiment/analyse/col_beam_70MeV.csv")
    # mylplotlib.plot_beam_dist(data[70], analyze.get_gfith(data[70], lim_fits[70]), lim_fits=lim_fits[70],
    #                             name="70", pdf=True, cmin=True)
    # print(data[200])
    # filts70 = [[i, j] for i, j in zip(col_beam_data70["paramsX"].values, col_beam_data70["paramsY"].values)]
    # filts200 = [[i, j] for i, j in zip(col_beam_data200["paramsX"].values, col_beam_data200["paramsY"].values)]
    # print(filts70)
    # mylplotlib.plot_exp_nclusters([data[70], data[200]], name="total", title="Total")
                                #   filts=[filts70, filts200], title=r"total")
    # prof70_data = pd.read_csv("./data/experiment/analyse/col_beam_70MeV.csv")
    # mylplotlib.plot_beam2d(prof70_data, lim=[lim_fits[70]["x"][-1], lim_fits[70]["y"][-1]],
    #                        name="", title="", bi=False)
    # mylplotlib.plot_HoverH2(data[70], name="col_beam_70MeV", title="70 MeV", zoom=[[350, 650], [125, 425]])
    # mylplotlib.plot_HoverH2(data[100], name="col_beam_100MeV", title="100 MeV", zoom=[[350, 650], [125, 425]])
    # mylplotlib.plot_HoverH2(data[120], name="col_beam_120MeV", title="120 MeV", zoom=[[350, 650], [125, 425]])
    # mylplotlib.plot_HoverH2(data[150], name="col_beam_150MeV", title="150 MeV", zoom=[[350, 650], [125, 425]])
    # mylplotlib.plot_HoverH2(data[180], name="col_beam_180MeV", title="180 MeV", zoom=[[350, 650], [125, 425]])
    # mylplotlib.plot_HoverH2(data[200][data[200].eventID < 600], name="col_beam_200MeV", title="200 MeV", zoom=[[350, 650], [125, 425]])
    # beam70_data = pd.read_csv("./data/experiment/data-col/datasub_70MeV1000MU.csv", index_col=None)
    # beam70_data_wo4 = beam70_data[(beam70_data.clusterSize > 1) & (beam70_data.layerID != 4)]
    # beam70_data_w4 = beam70_data[(beam70_data.clusterSize > 2) & (beam70_data.layerID == 4)]
    # mylplotlib.plot_HoverH4(pd.concat([beam70_data_w4, beam70_data_wo4], ignore_index=True),
    #                         energy=70, title="70 MeV")

#-------------------------- Ready for track reconstruction -------------------------#
#     data_t = pd.read_csv("./data/experiment/data_sel_2sigma_hits.csv")
#     data_layer0 = [
#         pd.read_csv("./data/experiment/reconstruction/e70_nhits0.csv"),
#         pd.read_csv("./data/experiment/reconstruction/e200_nhits0.csv"),
#     ] 
#     data_fail = pd.read_csv("./data/experiment/reconstruction/fail_hits.csv")
#     for e_i, e in enumerate([70, 200]):
#         smax = cfg['experiment'][f'{e} MeV']['smax']
#         events_0 = np.unique(data_layer0[e_i][data_layer0[e_i].nhits == 0]["eventID"].values)
#         # events_fail = np.unique(data_fail[data_fail.energy == e]["eventID"].values)
#         # events_none = np.unique(np.concatenate((events_0, events_fail), axis=None))
#         data_rec = data_t[data_t.energy == e]
#         # track_rec_list = []
#         for evt_i, evt in enumerate(np.unique(data_rec["eventID"].values)):
#             pd_tracks_list = []
#             for msc_i, msc in enumerate(np.linspace(cfg["const"]["msc_angle"][f"{e}"], 0.1, 100)):
#                 all_tracks_data = rec.run_rec(
#                     data_rec[data_rec.eventID == evt],
#                     msc,
#                     smax
#                 ) 
#                 pd_tracks_data = rec.collect_track_data(all_tracks_data)
#                 if all_tracks_data and 'MyTrackID' in pd_tracks_data.columns:
#                     # track_rec_list.append(all_tracks_data)
#                     pd_tracks_data.insert(0, "MSCangle", len(pd_tracks_data.index)*[msc])
#                     pd_tracks_list.append(pd_tracks_data)
#             print(f"Finished reconstruction on energy: {e}, event id: {evt}")
#             if pd_tracks_list:
#                 pd.concat(pd_tracks_list, ignore_index=True).\
#                     to_csv(f"./data/experiment/reconstruction/ALL/e{e}\
# /event_{'0'*(4 - len(str(evt))) + str(evt)}.csv", index=False)
            # mylplotlib.plot_rec_tracks(track_rec_list, kind="experiment", name=f'{e} MeV')
        
    # print(phys.cal_HL_angle(energy=70, m=938.272, len=9.308, d=0.01))

# ----------------- Correlation ---------------------------#
    # ALPIDE_coms = list(combinations([0, 1, 2, 3, 4, 5], 2))
    # data_track = pd.read_csv("./data/experiment/reconstruction/track_all.csv")
    # print(f"{data_track['posX'].min()}, {data_track['posX'].max()}")
    # print(f"{data_track['posY'].min()}, {data_track['posY'].max()}")
    # pd.DataFrame(analyze.find_correlation(data_track)).\
    #     to_csv("./data/experiment/reconstruction/correl.csv", index=False)
    # data_corl = pd.read_csv("./data/experiment/reconstruction/correl.csv")
    # mylplotlib.plot_corl(data_corl)
    
#------------------------------ R value -------------------------------#
    #data = pd.read_csv("./data/experiment/reconstruction/correlation/r_values.csv", index_col=None)
    #slope_err = mylplotlib.plot_rvalue_mcs(data)
    #pd.DataFrame(slope_err).to_csv("./output/data/slope05.csv", index=None)

#------------------------------ Plot cluster --------------------------------#
#     file_name = "2255.root"
#     file_path = "/home/arnon/Projects/telescope-track-\
# following-old/data/experiment/hits/stack/Col200MeV500MU/"
#     file_list = os.listdir(file_path)
#     file_list.sort()
#     cluster_tracks = []
#     hit_data = cluster.exproot2array(file_path + file_name)
    # print(np.shape(np.array(hit_data[0])))
    # hit_data_0 = np.array(hit_data[0])
    # clusters_data = cluster.get_clusters(file_path + file_name, lim=lim_dict["200 MeV"], cut_size=2)
    # mylplotlib.plot_HoverC(hit_data_0, clusters_data[0]
    # noise_path = "/home/arnon/Projects/telescope-track-following/data/experiment/BG/root/"
    # noise_root_files = os.listdir(noise_path)
    # noise_data = []
    # noise_cluster_data = {"eventID": [], "layerID": [], "clusterSize": []}
    # for nrf_i, nrf in enumerate(noise_root_files):
    #     root_data = uproot.open(f"{noise_path}{nrf}")
    #     noise_data.append(root_data['EUDAQMonitor/Hits vs Plane;1'].values())
    #     noise_clusters = cluster.get_clusters_bg(f"{noise_path}{nrf}")
    #     for l_i in range(len(noise_clusters)):
    #         if noise_clusters[l_i]:
    #             for c_i in range(len(noise_clusters[l_i])):
    #                 noise_cluster_data["eventID"].append(nrf_i)
    #                 noise_cluster_data["layerID"].append(l_i)
    #                 noise_cluster_data["clusterSize"].append(len(noise_clusters[l_i][c_i]))
    #     print(f"Finished file {nrf_i}/{len(noise_root_files)}")
    # noise_data = np.array(noise_data)
    # np.savetxt(os.path.join("/home/arnon/Projects/telescope-track-following/data/experiment/BG/csv", 
    #                         "pnoise.csv"), noise_data)
    # pd.DataFrame(noise_cluster_data).to_csv(os.path.\
    #     join("/home/arnon/Projects/telescope-track-following/data/experiment/BG/frame", "cnoise"),
    #     index=None)
    
    # pnoise_path = "/home/arnon/Projects/telescope-track-following/data/experiment/BG/csv/pnoise.csv"
    # pnoise_data = np.genfromtxt(pnoise_path)
    # mylplotlib.plot_pnoise(pnoise_data)
    # cnoise_path = "/home/arnon/Projects/telescope-track-following/data/experiment/BG/frame/cnoise.csv"
    # cnoise_data = pd.read_csv(cnoise_path, index_col=None)
    # print(cnoise_data)
    # mylplotlib.plot_cnoise(cnoise_data)
    # noise.get_hotpixel()
    # all_track_data = pd.read_csv("./data/experiment/reconstruction/track_all.csv", index_col=None)
    # mul_track_data70 = all_track_data[(all_track_data.energy == 70) &\
    #         (all_track_data.MSCangle == 0.0451313131313131)]
    # print(mul_track_data70.iloc[:3, :])
    # event50_70MeV = (np.unique(mul_track_data70["eventID"].values))[:50]
    # for i in range(len(all_track_indicies)):
        # all_track_indicies2 = all_track_indicies[i: i + 1]
        # mul_track_data70_50evt = mul_track_data70[mul_track_data70.eventID.isin(all_track_indicies2)]
        # mylplotlib.plot_rec_tracks(mul_track_data70_50evt, inpixel=True, scatter=True, name=f"coltrack70MeV_evt{i}_{i+1}")
    # for evt_i, evt in enumerate(event50_70MeV):
    #     mylplotlib.plot_rec_tracks(mul_track_data70[mul_track_data70.eventID.isin(event50_70MeV[:evt_i])], 
    #                                inpixel=True, scatter=True, name=f"coltrack70MeV_frame{'0'*(2 - len(str(evt_i)))}{evt_i}")
    # ALPIDE_coms = list(combinations([0, 1, 2, 3, 4, 5], 2))
    # print(ALPIDE_coms)
    # data_corl = pd.read_csv("./data/experiment/reconstruction/correl.csv")
    # data_70_corl = data_corl[(data_corl.energy == 70) & (data_corl.mcs == 0.0451313131313131)]
    # data_corl_70x = data_corl[(data_corl.energy == 70) & (data_corl.mcs == 0.0451313131313131)\
    #     & (data_corl.comb==4)]
    # data_corl_70y = data_corl[(data_corl.energy == 70) & (data_corl.mcs == 0.0451313131313131)\
    #     & (data_corl.comb==19)]
    # mylplotlib.plot_corl_optz(data_corl_70x, "x")
    # data_200_corl = data_corl[(data_corl.energy == 200) & (data_corl.mcs > 0.040) & (data_corl.mcs < 0.041)]
    # data_200_corl = data_corl[(data_corl.energy == 200) & (f"{data_corl.mcs: .5f}" == "0.040881")]
    # print(data_200_corl)
    # print(data_70_corl)
    # mylplotlib.plot_corl(data_200_corl)

    # data = pd.read_csv("./data/experiment/cluster_frame.csv", index_col=None)
    # data_col = utils.collect_col_chits(data)
    # data_c_70 = data_col[data_col.energy == 70]
    # data_c_200 = data_col[data_col.energy == 200]
    # mylplotlib.plot_exp_nclusters(pd.concat([data_c_70, data_c_200], ignore_index=True))
    # mylplotlib.plot_hist_cluster2([data_c_70["clusterSize"].values, 
    #                                 data_c_200["clusterSize"].values], 
    #                                 name="70_200_layer")
    # mylplotlib.plotavg_edep_and_cluster(data_col, cfg["simulation"])
    # for l in range(6):
    #     mylplotlib.plot_hist_cluster2([data_c_70[data_c_70.layerID == l]["clusterSize"].values, 
    #                                    data_c_200[data_c_200.layerID == l]["clusterSize"].values], 
    #                                   name=f"70_200_layer_{l}")
    # mylplotlib.plot_hist_cluster(data_c_70, name="70_MeV", title="")

    #------------------------------ Plot track MCS ----------------------------#
    # data_fail = pd.read_csv("./data/experiment/reconstruction/fail_hits.csv", index_col=None)
    # data0_70 = pd.read_csv("./data/experiment/reconstruction/ALL/e70/event_0005.csv", index_col=None)
    # data0_200 = pd.read_csv("./data/experiment/reconstruction/ALL/e200/event_0000.csv", index_col=None)
    # data0 = {70: data0_70, 200: data0_200}
    # for e_i, e in enumerate([70, 200]):
    #     eff_track = {}
    #     dpath = f"./data/experiment/reconstruction/ALL/e{e}"
    #     for mcs_i, mcs in enumerate(np.sort(np.unique(data0[e]['MSCangle'].values))):
    #         eff_track[mcs] = [0, 0]
    #         for f_i, f in enumerate(os.listdir(dpath)):
    #             if int(f[6:-4]) not in list(np.unique(data_fail[data_fail.energy == e]\
    #                 ['eventID'].values)):
    #                 data = pd.read_csv(os.path.join(dpath, f), index_col=None)
    #                 track_ids = np.sort(np.unique(data[data.MSCangle == mcs]['MyTrackID']))
    #                 for t_i, t in enumerate(track_ids):
    #                     if len(data[(data.MyTrackID == t) & (data.MSCangle == mcs)]) > 1:
    #                         eff_track[mcs][0] += 1
    #                     if len(data[(data.MyTrackID == t) & (data.MSCangle == mcs)]) == 6:
    #                         eff_track[mcs][1] += 1
    #         print(f"Finished MCS: {mcs} => {eff_track[mcs]}")
        # print(f"******** {eff_track} *************")
    #mylplotlib.plot_num_hot()

    # proj_18_data = proj.get_hit_from_proj(18)
    # mylplotlib.plot_HoverH(proj_18_data[0], hover=False, mul=True,
    #                         name="SingleProj18")
    # utils.roots2csv_projs("/home/arnon/workspace/KCMH-pCT", "all_projs")

    #-------------- Collecting data ------------------#
    # col_data_dir = {
    #     "70MeV1000MU": "./data/experiment/Col070MeV1000MU_merged",
    #     "100MeV1000MU": "./data/experiment/Col100MeV500MU_merged",
    #     "120MeV1000MU": "./data/experiment/Col120MeV500MU_merged",
    #     "150MeV1000MU": "./data/experiment/Col150MeV500MU_merged",
    #     "180MeV1000MU": "./data/experiment/Col180MeV500MU_merged",
    #     "200MeV1000MU": "./data/experiment/Col200MeV500MU_merged",
    #     }
    # for k, v in col_data_dir.items():
    #     utils.gen_roots2csv(v, k)

    # ---------------- Read col hit data ------------------#
    # col_hit_data = {
    #         kf[0]:pd.read_csv(kf[1], index_col=None)\
    #         for kf in [
    #             [70,  "./data/experiment/data-col/data_70MeV2000MU.csv"],
    #             [100, "./data/experiment/data-col/data_100MeV1000MU.csv"],
    #             [120, "./data/experiment/data-col/data_120MeV1000MU.csv"],
    #             [150, "./data/experiment/data-col/data_150MeV1000MU.csv"],
    #             [180, "./data/experiment/data-col/data_180MeV1000MU.csv"],
    #             [200, "./data/experiment/data-col/data_200MeV1000MU.csv"],
    #             ]
    #     }
    # def sub_offset(pos, layer_id, ke, dim):
    #     if dim == 'x':
    #         return pos - (cfg["experiment"]["70 MeV"]["mus"][dim][int(layer_id)] - 512)
    #     else:
    #         return pos - (cfg["experiment"]["70 MeV"]["mus"][dim][int(layer_id)] - 256)
    # col_hit_data_sub = col_hit_data.copy()
    # for ke in [70, 100, 120, 150, 180, 200]:
    # # for ke in [70]:
    #     col_hit_data_sub[ke].insert(len(col_hit_data[ke].columns), "posSubX", 
    #                             col_hit_data[ke]["posX"].values)
    #     col_hit_data_sub[ke]["posSubX"] = col_hit_data_sub[ke]\
    #         .apply(lambda row: sub_offset(row['posSubX'], row['layerID'], ke, 'x'), axis=1)

    #     col_hit_data_sub[ke].insert(len(col_hit_data[ke].columns), "posSubY", 
    #                             col_hit_data[ke]["posY"].values)
    #     col_hit_data_sub[ke]["posSubY"] = col_hit_data_sub[ke]\
    #         .apply(lambda row: sub_offset(row['posSubY'], row['layerID'], ke, 'y'), axis=1)

    #     col_hit_data_sub[ke].insert(len(col_hit_data[ke].columns), "cposSubX", 
    #                             col_hit_data[ke]["cposX"].values)
    #     col_hit_data_sub[ke]["cposSubX"] = col_hit_data_sub[ke]\
    #         .apply(lambda row: sub_offset(row['cposSubX'], row['layerID'], ke, 'x'), axis=1)

    #     col_hit_data_sub[ke].insert(len(col_hit_data[ke].columns), "cposSubY", 
    #                             col_hit_data[ke]["cposY"].values)
    #     col_hit_data_sub[ke]["cposSubY"] = col_hit_data_sub[ke]\
    #         .apply(lambda row: sub_offset(row['cposSubY'], row['layerID'], ke, 'y'), axis=1)

    #     col_hit_data_sub[ke].to_csv(f"./data/experiment/data-col/datasub_{ke}MeV1000MU.csv", index=None)
    # col_hit_data[70].insert(len(col_hit_data[70].columns), "posSubY")
    # col_hit_data[70].insert(len(col_hit_data[70].columns), "posSubCX")
    # col_hit_data[70].insert(len(col_hit_data[70].columns), "posSubCY")

    #------------- Analyze ------------------#
    # hit_data = pd.read_csv("./data/experiment/data-col/data_70MeV2000MU.csv", index_col=None)
    # hit_data = pd.read_csv("./newdata/datasubselchit_70MeV1000MU.csv", index_col=None)
    # data_hit_wo_noise_layer0 = hit_data[(hit_data.clusterSize > 1) & (hit_data.layerID == 0)]
    # data_hit_wo_noise_layer0 = data_hit_wo_noise_layer0["posSubX"].values 
    # data_hit_wo_noise_layer0.sort()

    # hit_data = hit_data.drop_duplicates(subset=["eventID", "hitID", "layerID"])
    # hit_data_wo_noise = hit_data[hit_data.clusterID > 1]
    # hit_data_wo_noise_posx_layer0 = np.sort(hit_data_wo_noise[hit_data_wo_noise.layerID == 0]["posX"].values)
    # values, counts = np.unique(data_hit_wo_noise_layer0, return_counts=True)
    # plt.plot(values, counts)
    # plt.show()
    # layer = 5
    # energy = 70
    # hit_data_sub = pd.read_csv("./data/experiment/data-col/datasub_{}MeV1000MU.csv".format(energy),
                            #    index_col=None)
    # hit_data_sub = hit_data_sub.drop_duplicates(subset=["clusterID"])
    # print(hit_data_sub[hit_data_sub.clusterSize > 1]["posSubX"])
    # analyze.get_hit_data(hit_data_sub[hit_data_sub.clusterSize > 1]["posX"].values)
    # analyze.get_hit_data(hit_data_sub[(hit_data_sub.clusterSize > 1) & (hit_data_sub.layerID == 0)]["posSubX"].values)
    # analyze.get_est_hit_data(
    #     hit_data_sub[(hit_data_sub.clusterSize > 1) & (hit_data_sub.layerID == layer)]["posSubX"].values,
    #     512, lim=cfg["experiment"]["{} MeV".format(energy)]["range"]["xsub"][layer]
    #     )
    
    #--------------- Fitting sub Guassian ----------------#
    # hit_data_sub = pd.read_csv("./data/experiment/data-col/data_{}MeV1000MU.csv".format(energy),
    #                            index_col=None)
    # pd_data_dict = {"energy": [], "axis": [], "section": [], "layer": [], "mu1": [], 
    #                 "sigma1": [], "amp1": [], "mu2": [], "sigma2": [], "amp2": []}
    # energies = [70, 100, 120, 150, 180, 200]
    # for energy_i, energy in enumerate(energies):
    #     hit_data = pd.read_csv("./data/experiment/data-col/data_{}MeV1000MU.csv".format(energy),
    #                             index_col=None)
    #     for axis_i, axis in enumerate(["x", "y"]):
    #         # sections = 3 if energy != 180 else 2
    #         sections = 2
    #         splited_hit_data = utils.split_data(hit_data, sections)
    #         for section in range(sections):
    #             layers = np.unique(splited_hit_data[section]["layerID"].values)
    #             layers.sort()
    #             for layer_i, layer in enumerate(layers):
    #                 params = []
    #                 if axis == "x":
    #                     lim = [0, 1024]
    #                     data_layer = splited_hit_data[section][(splited_hit_data[section].layerID == layer) &
    #                                             (splited_hit_data[section].clusterSize > 1) &
    #                                             (splited_hit_data[section].posX <= lim[1]) &
    #                                             (splited_hit_data[section].posX >= lim[0])]["posX"].values \
    #                                                 if layer != 4 else splited_hit_data[section][(splited_hit_data[section].layerID == layer) &
    #                                             (splited_hit_data[section].clusterSize > 2) &
    #                                             (splited_hit_data[section].posX <= lim[1]) &
    #                                             (splited_hit_data[section].posX >= lim[0])]["posX"].values
    #                     params = analyze.get_gfit2g(data_layer, expected=cfg["experiment"]["{} MeV".format(energy)]["expectedsub"]["x"][layer])
    #                 else:
    #                     lim = [0, 512]
    #                     data_layer = splited_hit_data[section][(splited_hit_data[section].layerID == layer) &
    #                                             (splited_hit_data[section].clusterSize > 1) &
    #                                             (splited_hit_data[section].posY <= lim[1]) &
    #                                             (splited_hit_data[section].posY >= lim[0])]["posY"].values \
    #                                                 if layer != 4 else splited_hit_data[section][(splited_hit_data[section].layerID == layer) &
    #                                             (splited_hit_data[section].clusterSize > 2) &
    #                                             (splited_hit_data[section].posY <= lim[1]) &
    #                                             (splited_hit_data[section].posY >= lim[0])]["posY"].values
    #                     params = analyze.get_gfit2g(data_layer, expected=cfg["experiment"]["{} MeV".format(energy)]["expectedsub"]["y"][layer])
    #                 pd_data_dict["energy"].append(energy)
    #                 pd_data_dict["axis"].append(axis)
    #                 pd_data_dict["section"].append(section)
    #                 pd_data_dict["layer"].append(layer)
    #                 pd_data_dict["mu1"].append("{:.2f}".format(params[0]))
    #                 pd_data_dict["sigma1"].append("{:.2f}".format(params[1]))
    #                 pd_data_dict["amp1"].append("{:.2f}".format(params[2]))
    #                 pd_data_dict["mu2"].append("{:.2f}".format(params[3]))
    #                 pd_data_dict["sigma2"].append("{:.2f}".format(params[4]))
    #                 pd_data_dict["amp2"].append("{:.2f}".format(params[5]))
    # pd_data = pd.DataFrame(pd_data_dict)
    # pd_data["energy"] = pd_data["energy"].astype("int")
    # pd_data["section"] = pd_data["section"].astype("int")
    # pd_data["layer"] = pd_data["layer"].astype("int")
    # pd_data["mu1"] = pd_data["mu1"].astype("float64")
    # pd_data["sigma1"] = pd_data["sigma1"].astype("float64")
    # pd_data["amp1"] = pd_data["amp1"].astype("float64")
    # pd_data["mu2"] = pd_data["mu2"].astype("float64")
    # pd_data["sigma2"] = pd_data["sigma2"].astype("float64")
    # pd_data["amp2"] = pd_data["amp2"].astype("float64")
    # pd_data.to_csv("./beamfit_section.csv", index=False)
    data = pd.read_csv("./beamfit_section.csv", index_col=None)
    data_hit = pd.read_csv("./newdata/datasub_70MeV1000MU.csv", index_col=None)
    # data_x_l0 = utils.select_center_fit(data, axis="x", layer=0,
    #                                     nosigs=[1.54])
    # data_y_l0 = utils.select_center_fit(data, axis="y", layer=0,
    #                                     nosigs=[-5.41, 2.55, 0.08, 24.76])
    # data_x_l1 = utils.select_center_fit(data, axis="x", layer=1,
    #                                     nosigs=[])
    # data_y_l1 = utils.select_center_fit(data, axis="y", layer=1,
    #                                     nosigs=[])
    # data_x_l2 = utils.select_center_fit(data, axis="x", layer=2,
    #                                     nosigs=[])
    # print(data_x_l2)
    # print(data[(data.layer == 0) & (data.axis == "y")])
    # mylplotlib.plot_box_center(data, axis="y")
    mylplotlib.plot_center_zscore(data, axis="y")
    # mylplotlib.plot_6hist_center_line(data_hit, axis="y")
        # print("{}".format(lim))
        # analyze.get_est_hit_data(
        #     data_layer,
        #     512, lim=lim
        #     )
    # plt.show()
    # beamfit_data = utils.read_beam_fit("./beamfitlogs.txt")
    # print(beamfit_data)
    # mylplotlib.plot_hit_events("./data/experiment/Col070MeV1000MU_merged")
    # splited_data = utils.split_data(hit_data_sub, 4)
    # print(splited_data)


    #-------------- reconstrution with subtracted ----------------------#
    # sub_data = pd.read_csv("./data/experiment/data-col/datasub_70MeV1000MU.csv", 
    #                                                  index_col=None)
    # evts = np.unique(sub_data["eventID"].values)
    # evts.sort()
    # sub_sel_chit_data = []
    # for evt_i, evt in enumerate(evts):
    #     sub_sel_chit_data2 = rec.sel_csubhit(rec.get_csubhit_csv(
    #         sub_data[sub_data.eventID == evt]
    #         ), energy="70")
    #     if not sub_sel_chit_data2.empty:
    #         sub_sel_chit_data.append(sub_sel_chit_data2)
    # pd.concat(sub_sel_chit_data, ignore_index=True)\
    #     .to_csv("./data/experiment/data-col/datasubselchit_70MeV1000MU_xxx.csv", index=False)

    #--------------- plot cpos ------------------#
    # mylplotlib.plot_6cpos(pd.read_csv("./data/experiment/data-col/datasubselchit_70MeV1000MU.csv",
    #                                   index_col=None), axis="y")

    #-------------- reconstruct sub data -------------------------#
    # for_track_data = pd.read_csv("./data/experiment/data-col/datasubselchit_70MeV1000MU.csv", index_col=None)
    # evts = np.unique(for_track_data["eventID"].values)
    # rec_track_list = [] 
    # evts.sort()
    # smaxs = np.arange(1, 200, 2)
    # mcss = np.arange(1, 100)
    # for smax in smaxs:
    #     for mcs in mcss:
    #         print(f"Starting Smax: {smax}, MCS: {mcs}")
    #         for evt_i, evt in enumerate(evts):
    #             for_track_data_evt = for_track_data[for_track_data.eventID == evt]
    #             rec_data = rec.run_rec(for_track_data_evt, mcs, smax)
    #             collected_track_data = rec.collect_track_data(rec_data)
    #             collected_track_data.insert(0, "mcs", [mcs]*len(collected_track_data.index))
    #             collected_track_data.insert(0, "smax", [smax]*len(collected_track_data.index))
    #             # print(collected_track_data)
    #             rec_track_list.append(collected_track_data)
    #         print(f"Finished Smax: {smax}, MCS: {mcs}\n")
    # pd.concat(rec_track_list, ignore_index=True).\
    #     to_csv("./data/experiment/reconstruction/sub/e70MeV", index=False)
    # mylplotlib.plot_alpide_grid_track(collected_track_data, energy=70)

    #---------------------- Fitting cluster --------------------------#
    # colsubhit_data = pd.read_csv("data/experiment/data-col/datasubselchit_70MeV1000MU.csv", index_col=None)
    # colsubhit_data_w_layer4 = colsubhit_data[(colsubhit_data.layerID == 4) & (colsubhit_data.clusterSize > 2)] 
    # colsubhit_data_wo_layer4 = colsubhit_data[(colsubhit_data.layerID != 4) & (colsubhit_data.clusterSize > 1)] 
    # colsubhit_data_wo_noise = pd.concat([colsubhit_data_w_layer4, colsubhit_data_wo_layer4], ignore_index=True)
    # colsubhit_data_wo_noise = colsubhit_data_wo_noise.sort_values(by=['eventID', 'layerID'])
    # mylplotlib.plot_avg_edep_and_cluster(colsubhit_data, cfg['simulation'])