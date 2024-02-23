import os
import numpy as np
from modules import cluster, utils

def collect_bgs():
    bg_path = "./data/experiment/BG/root/"
    bg_files = os.listdir(bg_path)
    bg_files.sort()
    bg_clusters = [
        cluster.get_clusters(bg_path + i, cut_size=0)\
            for i in bg_files
    ]
    bg_cluster_layers = []
    for i in range(6):
        bg_layer = []
        for j in range(len(bg_files)):
            if bg_clusters[j][i]:
                bg_layer += bg_clusters[j][i]
        bg_cluster_layers.append(bg_layer)
    # print(bg_cluster_layers)
    # for i in range(6):
    #     print(f"XXXXXXXXXXXXXX {i}")
    #     print(bg_cluster_layers[i])
    return bg_cluster_layers

def rename_bgroot(): 
    bg_path = "./data/experiment/BG/root/"
    bg_files = os.listdir(bg_path)
    for f in bg_files:
        os.rename(bg_path + f, bg_path + "0"*(4 - len(f[:-5])) + f)

def write_bgs():
    bg_path = "./data/experiment/BG/txt/"
    bg_data = collect_bgs()
    for i in range(len(bg_data)):
        data_len = [len(d) for d in bg_data[i]]
        max_len = max(data_len)
        np_data = np.ones((len(data_len), max_len))*(-1)
        for j in range(len(data_len)):
            for k in range(max_len):
                if k < len(bg_data[i][j]):
                    np_data[j, k] = bg_data[i][j][k]
                else:
                    break
        np.savetxt(bg_path + f"bg_layer{i}.txt", np_data)

# write_bgs()

def read_bgs(): 
    bg_path = "./data/experiment/BG/"
    bg_files = os.listdir(bg_path)
    bg_files.sort()
    bg_list = []
    for i in bg_files:
        bg_layer = []
        np_data = np.genfromtxt(bg_path + i)
        for j in np_data:
            bg_layer.append(j[j != -1].astype("int64").tolist())
        bg_list.append(bg_layer)
    return bg_list

def get_max_clusersize(bg_clusters, pixel, layer):
    cluster_len = []
    for c in bg_clusters[layer]:
        if pixel in c:
            cluster_len.append(len(c))
    return max(cluster_len)

def get_bg_pixels():
    data = read_bgs()
    data_npix = []
    for i in data:
        npix = []
        for j in i:
            for k in j:
                npix.append(k)
        data_npix.append(np.array(npix))
    bg_pixels = [utils.get_dead_pixel(np.array(d)) for d in data_npix]
    return bg_pixels

def remove_bg(hits_clusters):
    bg_pixels = get_bg_pixels()
    # print(bg_pixels[0])
    bg_clusters = read_bgs()
    for i in range(6):
        for c_exp_index in range(len(hits_clusters[i])):
            for p_bg in bg_pixels[i]:
                if p_bg in hits_clusters[i][c_exp_index]:
                    if len(hits_clusters[i][c_exp_index]) <=\
                        get_max_clusersize(bg_clusters, p_bg, i):
                            hits_clusters[i][c_exp_index] = []
                            break