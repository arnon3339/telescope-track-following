from modules import utils
import pandas as pd
import os
from os import path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

DATA_DIR = "./data/experiment/data-col/run56204405_240203204411"

def Gauss(x, A, x0, sigma):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

def Gauss_fit(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

def get_est_hit_data(data, mean, dis):
    data.sort()
    values, counts = np.unique(data, return_counts=True)
    pd_data = pd.DataFrame({"values": values, "counts": counts})
    print(pd_data)
    sel_pd_data = pd_data[(pd_data.values < dis[1]) & (pd_data.values > dis[0])]
    parameters, covariance = curve_fit(Gauss, 
                                       sel_pd_data["values"].values, 
                                       sel_pd_data["counts"].values,
                                       p0=[sel_pd_data["counts"].max(), mean, 15]
                                       )
    print(parameters)

if __name__ == "__main__":
    # x_range = [[300, 750]]
    # y_range = [[300, 750]]
    # x_mu = 512
    # y_mu = 256
    # proj_dirs = os.listdir(DATA_DIR)
    # files = os.listdir(DATA_DIR)
    # for f in files:
    #     fname = "0"*(5 - len(f.split('.')[0])) + str(int(f.split('.')[0])) + ".root"
    #     os.rename(path.join(DATA_DIR, f), path.join(DATA_DIR, fname))
    # data_path = "./data/experiment/data-col/run56170512_240203170517"
    # print(max(os.listdir(data_path)))
    data_path = DATA_DIR
    data = utils.gen_roots2csv(data_path, name="run56204405_240203204411")
    data = pd.read_csv("./data/experiment/data-col/data_run56170512_240203170517.csv", index_col=None)
    # data_layer_wo_4 = data[(data.layerID != 4) & (data.clusterSize > 1)]
    # data_layer_w_4 = data[(data.layerID == 4) & (data.clusterSize > 2)]
    # data_wo_noise = pd.concat([data_layer_wo_4, data_layer_w_4], ignore_index=True)
    # get_est_hit_data(data_wo_noise[data_wo_noise.layerID == 0]["posX"].values, x_mu, x_range[0])
    # plt.hist(data_wo_noise["posX"].values, 
    #          bins=range(data_wo_noise["posX"].min(), data_wo_noise["posX"].max(), 1)
    #          )
    # plt.xlim(x_range[0])
    # plt.show()