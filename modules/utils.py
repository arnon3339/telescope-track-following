import numpy as np
from ast import literal_eval
import os
from os import path
import uproot
import pandas as pd
import json
from modules import bg, cluster

with open('./config.json', 'r') as f:
    cfg = json.load(f)

def get_bg_data(bg_dir):
    data = []
    bg_files = os.listdir(bg_dir)
    bg_files.sort()
    bg_files = [i for i in bg_files if i[:5] == "BGrun"]
    for i in range(len(bg_files)):
        layer_data = []
        with open(bg_dir + bg_files[i], 'r') as fs:
            lines = fs.readlines()
            for line in lines:
                for elm in line.split(","):
                    elm2 = elm.replace('\n', '')
                    layer_data.append(int(elm2))
        data.append(layer_data)
    return data

def get_dead_pixel(data):
    data = np.unique(data, return_counts=True)
    print(np.sum(data[1]))
    indexs = np.array((np.where(data[1] > np.average(data[1])/2))[0])
    return [data[0][i] for i in indexs]

def get_rcpix(numpix):
    return [int(numpix/1024), int(numpix%1024)]

def edep2medep(edep):
    return (edep*1e3)/25

def edep2csize(edep):
    return np.ceil(4.23*(edep**0.65))
# print(data[0])

#--- Writing data to file path ----#
def collect_roothits(file_list, files_path):
    hist_keys = [
        'Hitmaps/ALPIDE_0/h_hitmap_ALPIDE_0;1',
        'Hitmaps/ALPIDE_1/h_hitmap_ALPIDE_1;1',
        'Hitmaps/ALPIDE_2/h_hitmap_ALPIDE_2;1',
        'Hitmaps/ALPIDE_3/h_hitmap_ALPIDE_3;1',
        'Hitmaps/ALPIDE_4/h_hitmap_ALPIDE_4;1',
        'Hitmaps/ALPIDE_5/h_hitmap_ALPIDE_5;1'
    ]
    data = []
    for i in range(len(file_list)):
        root_data = uproot.open(path.join(files_path, file_list[i]))
        hit_data = []
        try:
            for j in range(len(hist_keys)):
                hit_data.append(root_data[hist_keys[j]])
        except:
            continue
        data.append(hit_data)
    hist_layers = []
    for i in range(len(data[0])):
        hits_layer = []
        for j in range(len(data)):
            harr_data = np.where((data[j][i].to_numpy())[0].transpose() != 0)
            for k in range(len(harr_data[0])):
                hits_layer.append(np.array([harr_data[1][k], harr_data[0][k]]))
        hist_layers.append(np.array(hits_layer))
    return hist_layers

def get_colmon_tracks(sim_l0, sim):
    tracks = []
    for evt in sim_l0["eventID"].values:
        for trk in sim_l0[sim_l0.eventID == evt]["trackID"].values:
            track_data = sim[(sim.eventID == evt) & (sim.trackID == trk)].copy()
            track_data.sort_values(by=["layerID"])
            tracks.append(track_data)
    return tracks

def get_edep_colmon(tracks):
    edeps = []
    for trk in tracks:
        edeps += trk["edep"].values.tolist()
    return edeps

def count_cmpl_tracks(track_dir, dim=1):
    track_name = (track_dir.split(sep='/'))[-2]
    num_hit = np.genfromtxt(f"./output/{track_name}layer0.csv", dtype=np.int32)
    # num_hit = num_hit[:100]
    num_hit.astype(np.int32)
    f_list = [f"evt{i}.csv" for i in num_hit[:, 0]]
    chit = num_hit[:, 1]
    np_list = [np.genfromtxt(f"{track_dir}{f}", dtype=np.int32) for f in f_list]
    np_dict = {d: [] for d in np.unique(chit)}
    for d, arr, i, f in zip(chit.tolist(), np_list, range(len(np_list)), f_list):
        count = []
        # print(np.where(arr == 6)[0])
        # print(f"{np.shape(arr)}, {f}")
        if arr.ndim < 2 and not len(np.where(arr == 6)[0]):
            continue
        elif arr.ndim >= 2:
            for j in range(len(arr[0])):
                # print(np.shape(arr))
                if len(np.where(arr[:, j] == 6)[0]):
                    count.append(j)
            if len(count):
                new_arr = np.zeros((len(arr), len(count)), dtype=np.int32)
                if len(count) == 1:
                    new_arr = arr[:, count[0]]
                else:
                    for k in range(len(count)):
                        new_arr[:, k] = arr[:, count[k]]
                arr = new_arr
        np_dict[d].append(arr)
        # np.savetxt(f'./logs/npdict/e70Mev/{d}_{i}.csv', arr)
    # print(np_dict)
    return np_dict 

def get_smax_tracks(np_dict, many=0, nhit=2): 
    if many:
        pass
    else:
        data_dict = {i: [] for i in range(1, nhit + 1)}
        for i in range(1, nhit + 1):
            for j in range(len(np.linspace(0.2, 10, 50))):
                data = []
                for k in range(len(np_dict[i])):
                    if np_dict[i][k].ndim < 2:
                        data.append(np_dict[i][k][j])
                    else:
                        data += np_dict[i][k][j, :].tolist()
                data_dict[i].append(sum([1 for ii in data if ii == 6])/len(data))
        return data_dict
            
# def count_chit(chit_path, cmpl=True):
#     if cmpl:
#         pass
        
def root2csv(fpath, particles=""):
    root_data = uproot.open(fpath)
    hit_data = root_data["Hits;1"]
    keys = hit_data.keys()
    hit_dict = {k: hit_data[k].array().tolist() for k in keys}
    fname_out = (fpath.split(sep="/"))[-1][:-5]
    pd_data = pd.DataFrame(hit_dict)
    keys.remove("eventID")
    keys.remove("level1ID")
    keys.remove("trackID")
    keys.remove("edep")
    new_cols = ["eventID", "level1ID", "trackID", "edep"] + keys
    pd_data = pd_data.reindex(columns=new_cols)
    pd_data = pd_data.sort_values(by=["eventID", "trackID", "level1ID"])
    pd_data["trackID"] = pd_data["trackID"].astype('int64')
    if particles.lower() == "proton":
        pd_data = pd_data[pd_data.trackID == 1]
    elif particles.lower() == "secondary": 
        pd_data = pd_data[pd_data.trackID != 1]
    pd_data.to_csv(f"./data/simulation/csv/{fname_out}.csv")

def roothit2csv(data, name=""):
    data.to_csv(f"./data/simulation/csv/{name}{'_' if name else ''}roothit.csv")
    # for i in range(len(data_list)):
# def get_clusters_from_tracks(tracks, clusters):
#     for trk in tracks:
        
def merge_evnt_data(dirs=[]):
    dirs_path = "/".join((dirs[0].split(sep='/'))[:-1])
    target_dir_name = (dirs[0].split(sep='/'))[-1] + "_merged"
    if not os.path.exists(dirs_path + "/" + target_dir_name):
        os.mkdir(dirs_path + "/" + target_dir_name)    
    count = 0
    for i in range(len(dirs)):
        os.system(f"cp -r {dirs[i]}/* {dirs_path}/{target_dir_name}/")
        file_list = os.listdir(dirs[i])
        for j in range(len(file_list)):
            os.rename(dirs_path + "/" + target_dir_name + "/" + file_list[j],
                      dirs_path + "/" + target_dir_name +\
                          f"/event_{'0'*(4 - len(str(count + j))) + str(count + j)}.root")
        count = len(file_list)

def pdlist2list(str_list):  
    x = str_list.replace('   ', ' ')
    x = x.replace('  ', ' ')
    x = x.replace(' ', ',')
    return literal_eval(x)

def gen_2sigma_data(data_hits, data_beam): 
    data_hits = data_hits.copy()
    for e in np.unique(data_beam["energy"].values):
        for l in data_beam[data_beam.energy == e]["layerID"].values:
            meanX = data_beam[(data_beam.energy == e) & (data_beam.layerID == l)]["meanX"].values[0]
            meanY = data_beam[(data_beam.energy == e) & (data_beam.layerID == l)]["meanY"].values[0]
            sigmaX = data_beam[(data_beam.energy == e) & (data_beam.layerID == l)]["sigmaX"].values[0]
            sigmaY = data_beam[(data_beam.energy == e) & (data_beam.layerID == l)]["sigmaY"].values[0]
            lim = [[meanX - 2*sigmaX, meanX + 2*sigmaX], [meanY - 2*sigmaY, meanY + 2*sigmaY]]
            data_hits[e][data_hits[e].layerID == l] = data_hits[e][(data_hits[e].layerID == l) &                                        (data_hits[e].posX > lim[0][0]) & 
                                        (data_hits[e].posX > lim[0][0]) &
                                        (data_hits[e].posX < lim[0][1]) &
                                        (data_hits[e].posY > lim[1][0]) &
                                        (data_hits[e].posY < lim[0][1])]
        data_hits[e].dropna(subset=["eventID"], inplace=True)
        data_hits[e].insert(0, "energy", [e]*len(data_hits[e].index), True)
    pd.concat(data_hits, ignore_index=True).to_csv("./data/experiment/beam_data.csv", index=False)

def get_avg_pos(cluster_data):
    avg_x = sum([i%1024 for i in cluster_data])/len(cluster_data)
    avg_y = sum([int(i/1024) for i in cluster_data])/len(cluster_data)
    return (avg_x, avg_y)

def gen_roots2csv(files_path, name=""):
    data_dict = {'eventID':[], 'posX':[], 'posY': [],'layerID': [],
                 'clusterSize':[], 'hitID': [], 'clusterID': [], "cposX": [], "cposY": []}
    stack_files = os.listdir(files_path)
    stack_files.sort()
    stack_clusters = []
    for j in range(len(stack_files)):
        print(f"Starting {name} with event {j}")
        cluster_id = 0
        cluster_data = cluster.get_clusters(path.join(files_path, stack_files[j]))
        if cluster_data:
            stack_clusters.append(cluster_data)
            #bg.remove_bg(cluster_data)
            for k in range(len(cluster_data)):
                for kk in range(len(cluster_data[k])):
                    for kkk in range(len(cluster_data[k][kk])):
                        data_dict["hitID"].append(kkk)
                        data_dict["clusterID"].append(cluster_id)
                        data_dict["eventID"].append(j)
                        data_dict["clusterSize"].append(len(cluster_data[k][kk]))
                        data_dict["layerID"].append(k)
                        data_dict["posX"].append(cluster_data[k][kk][kkk]%1024)
                        data_dict["posY"].append(int(cluster_data[k][kk][kkk]/1024))
                        data_dict["cposX"].append(get_avg_pos(cluster_data[k][kk])[0])
                        data_dict["cposY"].append(get_avg_pos(cluster_data[k][kk])[1])
                    cluster_id += 1
        print(f"Finished {name} with event {j}")

    pd_merged_data = pd.DataFrame(data_dict)
    pd_merged_data["hitID"] = pd_merged_data["hitID"].astype('int')
    pd_merged_data["eventID"] = pd_merged_data["eventID"].astype('int')
    pd_merged_data["clusterSize"] = pd_merged_data["clusterSize"].astype('int')
    pd_merged_data["layerID"] = pd_merged_data["layerID"].astype('int')
    pd_merged_data["posX"] = pd_merged_data["posX"].astype('int')
    pd_merged_data["posY"] = pd_merged_data["posY"].astype('int')
    pd_merged_data["clusterID"] = pd_merged_data["clusterID"].astype('int')
    pd_merged_data.to_csv(f"./data/experiment/data-col/data_{name}.csv")

def roots2csv_projs(dir_path, name=""):
    data_dict = {'eventID':[], 'posX':[], 'posY': [],'layerID': [], 'proj': [],
                 'clusterSize':[], 'hitID': [], 'clusterID': [], "cposX": [], "cposY": []}
    dirs = os.listdir(dir_path)
    dirs.sort()
    for dir in dirs:
        files_path = path.join(dir_path, dir)
        stack_files = os.listdir(files_path)
        stack_files.sort()
        stack_clusters = []
        for j in range(len(stack_files)):
            print(f"Starting with projection {dir.split('_')[-1]} and file {stack_files[j]}")
            cluster_id = 0
            cluster_data = cluster.get_clusters(path.join(files_path, stack_files[j]), cut_size=0)
            if cluster_data:
                stack_clusters.append(cluster_data)
                # bg.remove_bg(cluster_data)
                for k in range(len(cluster_data)):
                    for kk in range(len(cluster_data[k])):
                        for kkk in range(len(cluster_data[k][kk])):
                            data_dict["hitID"].append(kkk)
                            data_dict["clusterID"].append(cluster_id)
                            data_dict["eventID"].append(j)
                            data_dict["clusterSize"].append(len(cluster_data[k][kk]))
                            data_dict["layerID"].append(k)
                            data_dict["proj"].append(dir.split('_')[-1])
                            data_dict["posX"].append(cluster_data[k][kk][kkk]%1024)
                            data_dict["posY"].append(int(cluster_data[k][kk][kkk]/1024))
                            data_dict["cposX"].append(get_avg_pos(cluster_data[k][kk])[0])
                            data_dict["cposY"].append(get_avg_pos(cluster_data[k][kk])[1])
                        cluster_id += 1
            print(f"Finished with projection {dir.split('_')[-1]} and file {stack_files[j]}")

        pd_merged_data = pd.DataFrame(data_dict)
        pd_merged_data["hitID"] = pd_merged_data["hitID"].astype('int')
        pd_merged_data["eventID"] = pd_merged_data["eventID"].astype('int')
        pd_merged_data["clusterSize"] = pd_merged_data["clusterSize"].astype('int')
        pd_merged_data["layerID"] = pd_merged_data["layerID"].astype('int')
        pd_merged_data["proj"] = pd_merged_data["proj"].astype('int')
        pd_merged_data["posX"] = pd_merged_data["posX"].astype('int')
        pd_merged_data["posY"] = pd_merged_data["posY"].astype('int')
        pd_merged_data["clusterID"] = pd_merged_data["clusterID"].astype('int')
        pd_merged_data.to_csv(f"./data/experiment/data_{name}.csv")

def get_hits_sigma(data, beam_data, area=2):
    data2 = []
    for i, layer in zip(range(len(np.unique(data["layerID"].values))), np.unique(data["layerID"].values)):
        data2.append(data[(data.layerID == layer) & ((data.cposX - beam_data["mus"]["x"][i])**2 +\
            (data.cposY - beam_data["mus"]["y"][i])**2 <=\
                (area*(beam_data["sigmas"]["x"][i] + beam_data["sigmas"]["y"][i])/2)**2)].copy())
        # print((area*(beam_data["sigmas"]["x"][i] + beam_data["sigmas"]["y"][i])/2)**2)
        # data2[i].to_csv(f"./logs/dd_{i}.csv")
    pd_data = pd.concat(data2, ignore_index=True)
    pd_data = pd_data.sort_values(by=["eventID", "layerID", "clusterID", "hitID"])
    # pd_data.to_csv("./logs/dd_comp.csv")
    return pd_data 
        # data2["x"].append(data[data[]])

def get_concE_data(data): 
    data_hits = data.copy()
    for e, v in data.items():
        data_hits[e].insert(0, "energy", [e]*len(data_hits[e].index), True)
    return pd.concat(data_hits, ignore_index=True)

def get_mean_en(data, kind="simulation"):
    return np.mean(data['edep'].values)

def get_rected_data(energies = [], filt=[[1, 2], [1, 2, 3, 4, 5, 6]]):
    forbid_evts = {"energy": [], "eventID": [], "nhits": []}
    nhits0_data = [
        pd.read_csv(f"./data/experiment/reconstruction/e{e}_nhits0.csv")\
            for e in energies
    ]
    data_dict = {e:{} for e in energies}
    for e_i in range(len(energies)):
        for f_i in range(len(filt[e_i])):
            d_evts = [d for d in nhits0_data[e_i][nhits0_data[e_i].nhits==filt[e_i][f_i]]["eventID"].values]
            file_names = [f"evt_{'0'*(4 - len(str(int(d)))) + str(int(d))}.csv"\
                for d in nhits0_data[e_i][nhits0_data[e_i].nhits==filt[e_i][f_i]]["eventID"].values]
            data_evts = [np.genfromtxt(f"./data/experiment/reconstruction/e{energies[e_i]}/{f}", delimiter=' ')\
                for f in file_names]
            invd_indexs = []
            # for devt_i in range(len(data_evts)):
            #     if not any(np.where(data_evts[devt_i] == 6)[0]):
            #         invd_indexs.append(devt_i)
            forbid_data = [d_evts[d_evt_i] for d_evt_i in invd_indexs]
            for df_i, df in enumerate(forbid_data):
                forbid_evts["eventID"].append(df)
                forbid_evts["nhits"].append(filt[e_i][f_i])
                forbid_evts["energy"].append(energies[e_i])
            # print(f"nhits0: {filt[e_i][f_i]}, nevt: {len(data_evts)}, nfevt: {len(invd_indexs)}")
            for index in sorted(invd_indexs, reverse=True):
                del data_evts[index] 
            data_dict[energies[e_i]][filt[e_i][f_i]] = data_evts
        pd.DataFrame(forbid_evts).to_csv("./data/experiment/reconstruction/forbid_events.csv", index=False)
    return data_dict

def get_forbid_hits(data, energies=[]):
    data_dict = []
    f_events = pd.read_csv("./data/experiment/reconstruction/forbid_events.csv")
    print(f_events.keys())
    for e_i, e in enumerate(energies):
        f_events_x = f_events[f_events.energy == e]["eventID"].values
        data_dict.append(data[(data.energy == e) & (data.eventID.isin(f_events_x))])
        print(data[(data.energy == e) & (data.eventID.isin(f_events_x))])
    data_dict = pd.concat(data_dict, ignore_index=True)
    return data_dict

def cal_highland_angle(kin, x, x0):
    return (14.1/(2*kin))*np.sqrt(x/x0)*(1 + (1/9)*np.log10(x/x0))

def cal_comp_x0(*args, **kwargs):
    """
    Calculate compound X0 by providing material properties in each layer 
    ordered by thickness, density, radiation length
    """
    data = np.array(args)
    r_val = np.sum([d[0]*d[1]/d[2] for d in data])
    return np.sum([d[0]*d[1] for d in data])/r_val

def collect_col_chits(data):
    data_out_list = []
    for e_i, e in enumerate(np.unique(data["energy"].values)):
        for l_i, l in enumerate(np.unique(data[data.energy == e]["layerID"].values)):
            data_t = data[(data.energy == e) & (data.layerID == l)]
            data_out_list.append(data_t[
                (data_t.cposX - cfg["experiment"][f"{e} MeV"]["mus"]["x"][l])**2 +
                (data_t.cposY - cfg["experiment"][f"{e} MeV"]["mus"]["y"][l])**2 <
                (2*(cfg["experiment"][f"{e} MeV"]["sigmas"]["x"][l] +\
                    cfg["experiment"][f"{e} MeV"]["sigmas"]["y"][l]))**2
                ])
    return pd.concat(data_out_list, ignore_index=True)

def read_beam_fit(file):
    data = {"energy": [], "axis": [], "layer": [], "mu1": [], "sigma1": [],
            "amp1": [], "mu2": [], "sigma2": [], "apm2": []}
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            if '#' in line:
                data_list1 = line.split(' ')
                f.readline()
                for layer in range(6):
                    data_list2 = f.readline().split('\t')
                    print(data_list2)
                    data["energy"].append(data_list1[1])
                    data["axis"].append(data_list1[3])
                    data["layer"].append(layer)
                    data["mu1"].append(data_list2[0])
                    data["sigma1"].append(data_list2[1])
                    data["amp1"].append(data_list2[2])
                    data["mu2"].append(data_list2[3])
                    data["sigma2"].append(data_list2[4])
                    data["apm2"].append(data_list2[5])
                line = f.readline()
    pd_data = pd.DataFrame(data)
    pd_data["energy"] = pd_data["energy"].astype("int")
    pd_data["layer"] = pd_data["layer"].astype("int")
    for column in pd_data.columns:
        if column not in ["energy", "axis", "layer"]:
            pd_data[column] = pd_data[column].astype("float64")
    return pd_data

def split_data(data: pd.DataFrame, n_part=1):
    data_len = len(data.index)
    d_data = int(data_len/n_part)
    new_data = []
    for i in range(n_part):
        if i != n_part -1:
            new_data.append(data.iloc[int(i*d_data): int((i + 1)*d_data), :])
        else:
            new_data.append(data.iloc[int(i*d_data):, :])
    return new_data

def select_center_fit(data: pd.DataFrame, axis="x", layer=0, 
                      nosigs=[]):
    new_data = data.copy()
    new_data = data[(data.axis==axis) & (data.layer==layer)]
    new_data = new_data[(~new_data.sigma1.isin(nosigs))]
    return new_data