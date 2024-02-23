import numpy as np
import uproot

def hit2vbool(root_data):
    data = np.array([False]*1024*512)
    hit_data = np.where(root_data > 0)
    v_data =[]
    for row, col in zip(hit_data[0], hit_data[1]):
        v_data.append(int(1024*row) + col)
    data[v_data] = True
    return data

def exproot2array(file_path, lim={}):
    X = np.array([range(0, 1024) for i in range(512)])
    Y = np.array([[i]*1024 for i in range(512)])
    data = []
    root_data =uproot.open(file_path)
    hit_keys = [f"Hitmaps/ALPIDE_{i}/h_hitmap_ALPIDE_{i};1" for i in range(6)]
    for key in hit_keys:
        if key in root_data.keys():
            data.append(root_data[key].to_numpy()[0].transpose())
    if lim and data:
        data[0][((X - lim["x"])**2)/lim["a"]**2 + ((Y - lim["y"])**2)/lim["b"]**2 > 1] = 0
    return data

def find_cluster(data=[]):
    clusters = []
    for i in range(len(data)):
        if data[i]:
            cluster = [i]
            data[i] = False
            find_sub_cluster(i, cluster, data)
            clusters.append(cluster)  
    return clusters

def find_sub_cluster(i, cluster, data):
    col = i%1024
    row = int(i/1024)
    
    if col - 1 >= 0 and data[1024*row + (col - 1)]:
        data[1024*row + (col - 1)] = False
        cluster.append(1024*row + (col - 1))
        find_sub_cluster(1024*row + (col - 1), cluster, data)
    
    if col + 1 <= 1023 and data[1024*row + (col + 1)]:
        data[1024*row + (col + 1)] = False
        cluster.append(1024*row + (col + 1))
        find_sub_cluster(1024*row + (col + 1), cluster, data)
        
    if row - 1 >= 0 and data[1024*(row - 1) + col]:
        data[1024*(row - 1) + col] = False
        cluster.append(1024*(row - 1) + col)
        find_sub_cluster(1024*(row - 1) + col, cluster, data)
    
    if row + 1 <= 511 and data[1024*(row + 1) + col]:
        data[1024*(row + 1) + col] = False
        cluster.append(1024*(row + 1) + col)
        find_sub_cluster(1024*(row + 1) + col, cluster, data)
    
    if col - 1 >= 0 and row - 1 >= 0 and data[1024*(row - 1) + (col - 1)]:
        data[1024*(row - 1) + (col - 1)] = False
        cluster.append(1024*(row - 1) + (col - 1))
        find_sub_cluster(1024*(row - 1) + (col - 1), cluster, data)
        
    if col - 1 >= 0 and row + 1 <= 511 and data[1024*(row + 1) + (col - 1)]:
        data[1024*(row + 1) + (col - 1)] = False
        cluster.append(1024*(row + 1) + (col - 1))
        find_sub_cluster(1024*(row + 1) + (col - 1), cluster, data)
    
    if col + 1 <= 1023 and row - 1 >= 0 and data[1024*(row - 1) + (col + 1)]:
        data[1024*(row - 1) + (col + 1)] = False
        cluster.append(1024*(row - 1) + (col + 1))
        find_sub_cluster(1024*(row - 1) + (col + 1), cluster, data)

    if col + 1 <= 1023 and row + 1 <= 511 and data[1024*(row + 1) + (col + 1)]:
        data[1024*(row + 1) + (col + 1)] = False
        cluster.append(1024*(row + 1) + (col + 1))
        find_sub_cluster(1024*(row + 1) + (col + 1), cluster, data)

    return

def get_HfromC(data):
    data = [d for d in data if d]
    arr_data = np.ones((len(data), 2))*(-1)
    for i in range(len(data)):
        arr_data[i, 0] = sum([j%1024 for j in data[i]])/len(data[i])
        arr_data[i, 1] = sum([int(j/1024) for j in data[i]])/len(data[i])
    return arr_data

def Cluster2Arr(data):
    data = [d for d in data if d]
    p_data = []
    for c in data:
        p_data += c
    arr_data = np.zeros((len(p_data), 2))
    for i in range(len(p_data)):
        arr_data[i, 0] = p_data[i]%1024
        arr_data[i, 1] = int(p_data[i]/1024)
    return arr_data

def get_clusters(file_path, cut_size=1, lim={}):
    root_hit_data = exproot2array(file_path, lim)
    root_vhit_data = [hit2vbool(data) for data in root_hit_data]
    #cut_sizes = [1, 1, 1, 1, 2, 1]
    cut_sizes = [0]*6
    return [cut_cluster_size(find_cluster(data), cut_sizes[data_i])\
        for data_i, data in enumerate(root_vhit_data)]

def get_clusters_bg(file_path):
    root_hit_data = exproot2array(file_path)
    root_vhit_data = [hit2vbool(data) for data in root_hit_data]
    return [cut_cluster_size(find_cluster(data), cut_size=0) for data in root_vhit_data]
   
def cut_cluster_size(clusters, cut_size):
    return [i for i in clusters if len(i) > cut_size]

def get_cluster_only(pd_data):
    pd_data2 = pd_data.drop_duplicates(subset=["energy", "eventID", "layerID", "clusterID"]).copy()
    return pd_data2.drop(columns=['posX', 'posY', 'hitID'])