import os
from os import path

from modules import utils

DATA_DIR = "/home/arnon/workspace/KCMH-pCT"

def get_hit_from_proj(proj):
    proj_dirs = os.listdir(DATA_DIR)
    for dir in proj_dirs:
        if int(proj) == int((dir.split('_')[-1]).split('.')[0]):
            files = os.listdir(path.join(DATA_DIR, dir))
            files.sort()
            return utils.collect_roothits(files, path.join(DATA_DIR, dir))
    return []

def get_cluster_from_proj(proj):
    pass

def get_all_clusters():
    pass

def get_all_hits():
    pass

# proj_dirs = os.listdir(DATA_DIR)

# for dir in proj_dirs:
#     files = os.listdir(path.join(DATA_DIR, dir))
#     for f in files:
#         fname = "0"*(5 - len(f.split('.')[0])) + str(int(f.split('.')[0])) + ".root"
#         os.rename(path.join(DATA_DIR, dir, f), path.join(DATA_DIR, dir, fname))