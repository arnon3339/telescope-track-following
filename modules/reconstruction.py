import numpy as np
import uproot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import random
import timeit
import os
import pandas as pd
import os
import json

with open("./config.json", 'r') as f:
    cfg = json.load(f)

def get_chit_data(cdata, cluster): 
    data_arr = np.array([i for i in cdata], dtype=object)
    temp_pd_data = pd.DataFrame(columns=["posX", "posY", "posZ", "layerID"])
    x_pos, y_pos, z_pos = [], [], []
    csize = []
    for i in range(len(data_arr)):
        for j in range(len(data_arr[i])):
            x_pos.append(data_arr[i][j, 0])
            y_pos.append(data_arr[i][j, 1])
            z_pos.append(i)
            csize.append(len(cluster[i][j]))
    temp_pd_data["posX"] = x_pos
    temp_pd_data["posY"] = y_pos
    temp_pd_data["posZ"] = z_pos
    temp_pd_data["posX"] = temp_pd_data["posX"]*30.0/1024 #mm
    temp_pd_data["posY"] = temp_pd_data["posY"]*13.8/512 #mm
    temp_pd_data["posZ"] = temp_pd_data["posZ"]*25.0 + 50 #mm
    temp_pd_data["cSize"] = np.array(csize)
    temp_pd_data["layerID"] = z_pos 
    temp_pd_data["ID"] = temp_pd_data.index
    temp_pd_data["layerID"] = temp_pd_data["layerID"].astype("int")
    temp_pd_data["ID"] = temp_pd_data["ID"].astype("int")
    return temp_pd_data

def get_chit_csv(csv_data):
    pd_layer_data_list = []
    data = csv_data.copy()
    for layer_i, layer in enumerate(np.unique(data["layerID"].values)):
        pd_layer_data = data[data.layerID == layer].copy()
        pd_layer_data_list.append(pd_layer_data.drop_duplicates(subset=["clusterID"]))
    res_data = pd.concat(pd_layer_data_list, ignore_index=True)
    res_data["ID"] = res_data.index
    res_data["layerID"] = res_data["layerID"].astype("int")
    res_data["posX"] = res_data["cposX"]*30.0/1024
    res_data["posY"] = res_data["cposY"]*13.8/512
    res_data["posZ"] = res_data["layerID"]*25.0 + 50
    return res_data

def get_csubhit_csv(csv_data):
    pd_layer_data_list = []
    data = csv_data.copy()
    for layer_i, layer in enumerate(np.unique(data["layerID"].values)):
        pd_layer_data = data[data.layerID == layer].copy()
        pd_layer_data_list.append(pd_layer_data.drop_duplicates(subset=["clusterID"]))
    res_data = pd.concat(pd_layer_data_list, ignore_index=True)
    res_data["ID"] = res_data.index
    res_data["layerID"] = res_data["layerID"].astype("int")
    res_data["posX"] = res_data["cposSubX"]
    res_data["posY"] = res_data["cposSubY"]
    res_data["posZ"] = res_data["layerID"]
    return res_data

def sel_csubhit(csubhit: pd.DataFrame, energy):
    layer_data = []
    for layer_i, layer in enumerate(np.unique(csubhit["layerID"].values)):
        x_2sigma = 2*cfg["experiment"][f"{energy} MeV"]["sigmas"]["xsub"][int(layer)]
        y_2sigma = 2*cfg["experiment"][f"{energy} MeV"]["sigmas"]["ysub"][int(layer)]
        layer_data.append(csubhit[(csubhit.layerID == layer) & 
                                  (csubhit.posX <= 512 + x_2sigma) &
                                  (csubhit.posX >= 512 - x_2sigma) &
                                  (csubhit.posY <= 256 + y_2sigma) &
                                  (csubhit.posY >= 256 - y_2sigma)
                                  ]
                          )
    return pd.concat(layer_data, ignore_index=True)
 
def cal_smax(n_p):
    return 0.047*n_p**(-0.176)

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.first_child = None
        self.parent = None
        
    def __repr__(self):
        return self.data

    def __str__(self):
        return self.data
        
class LinkList:
    def __init__(self, nodes=None, parent=None):
        self.head = None
        if nodes is not None:
            node = Node(data=nodes.pop(0))
            node.parent = parent
            self.head = node
            for elem in nodes:
                node.next = Node(data=elem)
                node = node.next
                node.parent = parent
        
    def __iter__(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next
        
    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(node.data)
            node = node.next
        nodes.append("None")
        return " -> ".join(nodes)
    
    def add_last(self, node):
        if not self.head:
            self.head = node
            return
        for current_node in self:
            pass
        current_node.next = node
        
def cal_theta0(energy):
    tua = energy/938.272
    pv = energy*(tua + 2)/(tua + 1)
    x = 25.0
    X0 = 8.897
    return 14.1*np.log(np.sqrt(x/X0))*(1 + (np.log(x/X0))/9)/pv

def cal_beta():
    sigma = cal_theta0(250)
    mu = 0
    np.random.seed(random.getrandbits())
    u1 = np.random.uniform()
    u2 = np.random.uniform()
    mag = sigma*np.sqrt(-2.0*np.log(u1))
    z0 = mag*np.cos(2.0*np.PI*u2) + mu
    z1 = mag*np.sin(2.0*np.PI*u2) + mu
    
    return np.sqrt(z0**2 + z1**2)

def cal_phi(z, alpha, beta):
    a = z*np.cos(beta - alpha) + z*np.cos(alpha + beta)
    x = a/2 - z*np.cos(alpha + beta)
    
def cal_new_pos(old_x, old_y, old_z, alpha, sigma):
    x1 = -old_z*np.sin(alpha) + old_x*np.cos(alpha)
    y1 = old_y
    z1 = old_z*np.cos(alpha) + old_x*np.sin(alpha)
    x2 = x1
    y2 = -z1*np.sin(sigma) + y1*np.cos(sigma)
    z2 = z1*np.cos(sigma) + y1*np.sin(sigma)
    return x2, y2, z2

def cal_angles(points, dalpha=0, dsigma=0):
    x, y, z = points["x"][1] - points["x"][0], points["y"][1] - points["y"][0], points["z"][1] - points["z"][0]
    new_x, new_y, new_z = cal_new_pos(x, y, z, dsigma)
    theta = np.arctan(np.sqrt(new_x**2 + new_y**2)/new_z)
    # with open("log.csv", mode="a") as file:
    #     spamwriter = csv.writer(file, delimiter=' ',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     spamwriter.writerow([points["x"][0], points["x"][1], points["y"][0], points["y"][1], points["z"][0], points["z"][1]])
    #     spamwriter.writerow([x, y, z, new_x, new_y, new_z])
    #     spamwriter.writerow([dalpha, theta, np.arctan(np.sqrt(x**2 + y**2)/z), np.arctan(y/x)])
    #     spamwriter.writerow("\n")
    return theta - dalpha, np.arctan(np.sqrt(x**2 + y**2)/z), np.arctan(y/x)

def cal_dvector(points):
    dvec = np.array([points["x"][1] - points["x"][0], points["y"][1] - points["y"][0], points["z"][1] - points["z"][0]])
    return dvec

def cal_dtheta(dvector, points):
    # d = np.sqrt(sum([(points["x"][1] - points["x"][0])**2 + (points["y"][1] - points["y"][0])**2 + (points["z"][1] - points["z"][0])**2]))
    m0m1 = np.array([points["x"][1] - points["x"][0], points["y"][1] - points["y"][0], points["z"][1] - points["z"][0]])
    cos_value = m0m1.dot(dvector)/\
        (np.sqrt(sum([np.power(elem, 2) for elem in m0m1])) * np.sqrt(sum([np.power(elem, 2) for elem in dvector])))
    # r = np.sqrt(cross.dot(cross))/np.sqrt(dvector.dot(dvector))
    if cos_value > 1:
        cos_value = 1
    elif cos_value < -1:
        cos_value = -1
    return np.fabs(np.arccos(cos_value))

def cal_S_n(S_n, theta):
    return np.sqrt(S_n**2 + theta**2)

def scan_track(data_proton_hits, node, removal_list, S_max, msc_angle):
    find_can_track(data_proton_hits, node, removal_list, S_max, msc_angle)
    if not node.next and not node.first_child:
        return
    elif node.first_child:
        scan_track(data_proton_hits, node.first_child, removal_list, S_max, msc_angle)
    elif node.next:
        scan_track(data_proton_hits, node.next, removal_list, S_max, msc_angle)
    elif node.parent:
        while not node.parent.next and node.parent:
            node = node.parent
        if node.parent.next:
            scan_track(data_proton_hits, node.parent.next, removal_list, S_max, msc_angle)
        else:
            return
    else:
        return

def find_can_track(pd_data_proton_hits, node, removal_list, S_max, msc_angle):
    if node.data["layerID"] == pd_data_proton_hits["layerID"].max():
        return
    is_first_Sn = True
    S_n = node.data["S_n"]
    dtheta, new_alpha, new_theta = 0, 0, 0
    next_hits = pd_data_proton_hits[pd_data_proton_hits["layerID"] == node.data["layerID"] + 1]
    child_list = []
    if not node.parent:
        node.data["S_n"] = 0
        node.data["dvector"] = np.array([0, 0, 1])
        node.data["theta"] = 0
    min_theta = 1e17
    for index, next_hit in next_hits.iterrows():
        points = {"x":[node.data["posX"], next_hit["posX"]], 
                  "y":[node.data["posY"], next_hit["posY"]], 
                  "z":[node.data["posZ"], next_hit["posZ"]]}
        dtheta = cal_dtheta(node.data["dvector"], points)
        temp_S_n = cal_S_n(S_n, dtheta)
        if dtheta < msc_angle:
            # print(dtheta)
            if S_max > temp_S_n and is_first_Sn:
                next_hit["S_n"] = temp_S_n
                next_hit["dvector"] = cal_dvector(points)
                next_hit["theta"] = dtheta
                child_list.append(next_hit)
                is_first_Sn = False
                min_theta = dtheta
            elif dtheta < min_theta and S_max > temp_S_n:
                # print(dtheta)
                next_hit["S_n"] = temp_S_n
                next_hit["dvector"] = cal_dvector(points)
                next_hit["theta"] = dtheta
                child_list.clear()
                child_list.append(next_hit)
                min_theta = dtheta
        # elif S_n == temp_S_n and S_max > temp_S_n:
        #     next_hit["S_n"] = S_n
        #     next_hit["dvector"] = cal_dvector(points)
            # child_list.append(next_hit)
    if child_list:
        for i in child_list:
            removal_list.add(int(i["ID"]))
        node.first_child = LinkList(child_list, node).head

class my_plot:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 15))
        self.ax = Axes3D(self.fig)
        self.ax.view_init(elev=20, azim=-30)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
    def show(self):
        self.fig.show()
    def save(self):
        self.ax.savefig("plot.jpg")
        
def plot_node(axx, node):
    if not node.first_child and not node.next:
        axx.scatter(node.data["posX"], node.data["posZ"], node.data["posY"], color='r')
        return
    elif node.first_child:
        axx.scatter(node.data["posX"], node.data["posZ"], node.data["posY"], color='r')
        axx.plot3D([node.data["posX"], node.first_child.data["posX"]], [node.data["posZ"], node.first_child.data["posZ"]], [node.data["posY"], node.first_child.data["posY"]])
        axx.scatter(node.first_child.data["posX"], node.first_child.data["posZ"], node.first_child.data["posY"], color='r')
        plot_node(axx, node.first_child)
    elif node.next:
        axx.scatter(node.next.data["posX"], node.next.data["posZ"], node.next.data["posY"], color='r')
        axx.plot3D([node.parent.data["posX"], node.next.data["posX"]], [node.parent.data["posZ"], node.next.data["posZ"]],
                   [node.parent.data["posY"], node.next.data["posY"]])
        plot_node(axx, node.next)
    elif node.parent:
        while not node.parent.next and node.parent:
            node = node.parent
        if node.parent.next:
            axx.scatter(node.parent.next.data["posX"], node.parent.next.data["posZ"], node.parent.next.data["posY"], color='r')
            axx.plot3D([node.parent.next.data["posX"], node.parent.next.parent.data["posX"]], [node.parent.next.data["posZ"], node.parent.next.parent.data["posZ"]],
                   [node.parent.next.data["posY"], node.parent.next.parent.data["posY"]])
            plot_node(node.parent.next)
        else:
            return
    else:
        return
        
def get_track(all_track, seed_node, track_list):
    # all_track.clear()
    temp_seed = seed_node
    while temp_seed.parent:
        track_list = [temp_seed] + track_list
        temp_seed = temp_seed.parent
    track_list = [temp_seed] + track_list
    all_track.append(track_list)

def collect_track(all_track, seed_node, track_list):
    if not seed_node.first_child and not seed_node.parent:
        return
    elif seed_node.first_child:
        collect_track(all_track, seed_node.first_child, track_list)
    elif not seed_node.first_child and seed_node.parent:
        get_track(all_track, seed_node, track_list)
        if seed_node.next:
            collect_track(all_track, seed_node.next, track_list)
        else:
            temp_seed = seed_node
            while temp_seed.parent:
                if temp_seed.parent.next:
                    collect_track(all_track, temp_seed.parent.next, track_list)
                temp_seed = temp_seed.parent
    else:
        return
    
def run_rec(data, msc_angle, S_max):
    data_proton_hits = data
    all_track = []
    # S_max = n*cal_smax(100)
    max_first = data_proton_hits[data_proton_hits["layerID"] == 0].copy()
    for index, row in max_first.iterrows():
        seed_track = []
        node = Node(row)
        node.data["S_n"] = 0
        removal_list = set([int(node.data["ID"])])
        seed_node = node
        scan_track(data_proton_hits, node, removal_list, S_max, msc_angle)
        collect_track(all_track, seed_node, seed_track)
        for i in removal_list:
            data_proton_hits = data_proton_hits[data_proton_hits.ID != i]
    return all_track

def collect_track_data(all_track):
    for i in range(len(all_track)):
        for j in range(len(all_track[i])):
            all_track[i][j].data["MyTrackID"] = int(i)
    s_data = []
    for i in range(len(all_track)):
        for j in range(len(all_track[i])):
            s_data.append(all_track[i][j].data)
    # s_data.reset_index(drop=True, inplace=True)
    # s_data = s_data.loc[:, ~s_data.columns.str.contains('^Unnamed')]
    if s_data:
        s_data = pd.DataFrame(s_data)
        if 'dvector' in s_data.columns:
            del s_data["dvector"]
        return s_data
    else:
        return pd.DataFrame({}) 

def plot_all_track(all_track, line=0):
    fig = plt.figure(figsize=(20, 15))
    ax = Axes3D(fig)
    if line:
        for i in all_track:
            ax.plot3D([j.data["posX"] for j in i], [j.data["posZ"] for j in i], [j.data["posY"] for j in i],
                    linewidth=2, c='black')
    else:
        for i in all_track:
            ax.scatter([j.data["posX"] for j in i], [j.data["posZ"] for j in i], [j.data["posY"] for j in i])
            ax.plot3D([j.data["posX"] for j in i], [j.data["posZ"] for j in i], [j.data["posY"] for j in i])
    ax.view_init(elev=20, azim=-10)
    # view_init_list = [[0, 0], [0, -30]]
    ax.set_xlabel("X mm")
    ax.set_ylabel("Z mm")
    ax.set_zlabel("Y mm")
    plt.show()
    
def collect_as_atrack_mon(mon_data):
    count_track = 0
    count_track2 = 0
    mon_tracks_dict = {}
    first_row = mon_data[mon_data["layerID"] == 0]
    for index, row in first_row.iterrows():
        other_hits = mon_data[(mon_data["trackID"] == row["trackID"]) & (mon_data["eventID"] == row["eventID"])]
        mon_tracks_dict[count_track] = {"layerID": [], "ID": [], "eventID": []}
        for index2, row2 in other_hits.iterrows():
            mon_tracks_dict[count_track]["layerID"].append(row2["layerID"])
            mon_tracks_dict[count_track]["ID"].append(row2["ID"])
            mon_tracks_dict[count_track]["eventID"].append(row2["eventID"])
        count_track += 1
    mon_tracks_dict_temp = mon_tracks_dict.copy()
    for key, value in mon_tracks_dict_temp.items():
        if len(value["layerID"]) == 1:
            del mon_tracks_dict[key]
    real_mon_tracks_dict = {}
    for key, value in mon_tracks_dict.items():
        real_mon_tracks_dict[count_track2] = value
        count_track2 += 1
    return real_mon_tracks_dict

def collect_as_atrack_rec(rec_data):
#     print(rec_data)
    count_track = 0
    count_track2 = 0
    rec_tracks_dict = {}
    if not len(rec_data.index):
        return {}
    first_row = rec_data[rec_data["layerID"] == 0].copy()
    for index, row in first_row.iterrows():
        other_hits = rec_data[(rec_data["MyTrackID"] == row["MyTrackID"]) & (rec_data["eventID"] == row["eventID"])]
        rec_tracks_dict[count_track] = {"layerID": [], "ID": [], "eventID": []}
        for index2, row2 in other_hits.iterrows():
            rec_tracks_dict[count_track]["layerID"].append(row2["layerID"])
            rec_tracks_dict[count_track]["ID"].append(row2["ID"])
            rec_tracks_dict[count_track]["eventID"].append(row2["eventID"])
        count_track += 1
    rec_tracks_dict_temp = rec_tracks_dict.copy()
    for key, value in rec_tracks_dict_temp.items():
        if len(value["layerID"]) == 1:
            del rec_tracks_dict[key]
    real_rec_tracks_dict = {}
    for key, value in rec_tracks_dict.items():
        real_rec_tracks_dict[count_track2] = value
        count_track2 += 1
    return real_rec_tracks_dict

def get_track_eff(rec_data, mon_data):
    count2 = 0
    rec_tracks = collect_as_atrack_rec(rec_data)
    mon_tracks = collect_as_atrack_mon(mon_data)
    if rec_tracks:
        for rec_key, rec_value in rec_tracks.items():
            ref_value = 0
            count1 = 0
            mon_ref_track = {}
            if rec_value["ID"]:
                if mon_tracks:
                        for mon_key, mon_value in mon_tracks.items():
                            if mon_value["eventID"][0] == rec_value["eventID"][0]:
                                mon_ref_track = mon_value
                                break
                if not mon_ref_track:
                    continue
                if len(rec_value["ID"]) > len(mon_ref_track["ID"]):
                    ref_value = len(rec_value["ID"])
                    for j in range(len(rec_value["ID"])):
                        for k in range(len(mon_ref_track["ID"])):
                            if rec_value["ID"][j] == mon_ref_track["ID"][k]:
                                count1 += 1
                else:
                    ref_value = len(mon_ref_track["ID"])
                    for j in range(len(mon_ref_track["ID"])):
                        for k in range(len(rec_value["ID"])):
                            if mon_ref_track["ID"][j] == rec_value["ID"][k]:
                                count1 += 1
                count2 += 1 if count1/ref_value >= 0.75 else 0     
    return count2/len(mon_tracks)

def cal_eff_in_smax(data, output=""):
    smax_eff_data = []
    count = 0
    for i in np.linspace(0.1, 10.1, 201):
        rec_data = collect_track_data(run_rec(data.copy(), i))
        smax_eff_data.append([i, get_track_eff(rec_data, data.copy())])
        print(smax_eff_data[count])
        count += 1 
    np.savetxt(f"./output/{output}", np.array(smax_eff_data), fmt=".4f")

def gen_rec_mulexp_mulsmax(data, energies=[]):
    # data = pd.read_csv("./data/experiment/data_2sigma_sel_hits.csv")
    event_ids = {e:np.unique(data[data.energy==e]["eventID"].values) for e in energies}
    smaxs = np.linspace(0.2, 20, 100)
    data_rec_dict = {e:{} for e in energies}
    rec_log = []
    for energy in energies:
        for evt in event_ids[energy]:
            tracks = []
            for smax in smaxs:
                rec_data = run_rec(data[(data.energy == energy) &\
                    (data.eventID == evt)], smax)
                rec_data_pd = collect_track_data(rec_data) 
                if 'MyTrackID' in rec_data_pd.columns:
                    count_tracks = rec_data_pd.groupby(['MyTrackID'])['MyTrackID'].count()
                    tracks.append(count_tracks.values.tolist())
                else:
                    tracks.append([0])
            print(f"energy: {energy}, event id: {evt}")
            rec_log.append(f"energy: {energy}, event id: {evt}")
            data_rec_dict[energy][evt] = tracks

    for e_k, e_v in data_rec_dict.items():
        evt_nhits = np.zeros((len(e_v), 2))
        e_dir = f"e{e_k}"
        for (evt_k, evt_v), ev_i in zip(e_v.items(), range(len(e_v))):
            evt_nhits[ev_i, 0] = evt_k
            evt_nhits[ev_i, 1] = len(data[(data.energy == e_k) & (data.eventID == evt_k)\
                & (data.layerID == 0)]["ID"].values)
            evt_path = f"evt_{'0'*(4 - len(str(evt_k))) + str(evt_k)}"
            max_arr_len = max([len(d) for d in evt_v])
            new_arr = np.zeros((smaxs.size, max_arr_len))
            for arr_r in range(len(data_rec_dict[e_k][evt_k])):
                new_arr[arr_r, :len(data_rec_dict[e_k][evt_k][arr_r])] = data_rec_dict[e_k][evt_k][arr_r]
            np.savetxt(f"./data/experiment/reconstruction/{e_dir}/{evt_path}.csv", new_arr)
        pd.DataFrame({"eventID": evt_nhits[:, 0], "nhits": evt_nhits[:, 1]}).\
            to_csv(f"./data/experiment/reconstruction/{e_dir}_nhits0.csv", index=False)
        # np.savetxt(f"./data/experiment/reconstruction/{e_dir}_nhits.csv", evt_nhits)   
# cal_eff_in_smax()

# proton_events = [f"{i*10}" for i in range(1, 10)]
# proton_events += [f"{i*100}" for i in range(1, 10)]
# proton_events += [f"{i*1000}" for i in range(1, 5)]

# for i in proton_events:
#     print(f"primary events = {i}")
#     cal_eff_in_smax(float(i))

# rec_data = collect_track_data(run_rec(4))
# mon_data_dict = collect_as_atrack_mon(get_hit_data(file_simulation, num_events))
# print("mon tracks:")
# for key in mon_data_dict:
#     print(f"{key}: {mon_data_dict[key]}")
    
# rec_data_dict = collect_as_atrack_rec(rec_data)
# print("rec tracks")
# for key in rec_data_dict:
#     print(f"{key}: {rec_data_dict[key]}")
# plot_scatter(get_chit_data())