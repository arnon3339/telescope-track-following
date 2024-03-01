from ast import literal_eval
import os
from os import path
from itertools import combinations
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.colors import ListedColormap,LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from modules import cluster 
from matplotlib.patches import Ellipse
from modules import utils
import matplotlib.mlab as mlab
from scipy.stats import norm, multivariate_normal, zscore
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib import ticker
from modules.analyze import Gauss, func_sigd, get_gfith, gauss
from scipy.optimize import curve_fit
from scipy import stats
import numpy.typing as npt
from typing import Any
import json

with open("./config.json", 'r') as f:
    cfg = json.load(f)
OUTPUT_DIR = "./output"
TIMES_FONT_PATH = r"./font/Times-New-Roman/"
# TIMES_REG = font_manager.FontProperties(TIMES_FONT_PATH + r"times new roman.ttf")
TIMES_BOLD = font_manager.FontProperties(fname=TIMES_FONT_PATH + r"times-new-roman-bold.ttf")
FNAME = TIMES_FONT_PATH + r"times-new-roman-bold.ttf"

FONT_SIZE = 28

def plot_scatter(data_hits):
    fig = plt.figure(figsize=(20, 16))
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=-30)
    ax.scatter(data_hits.posX, data_hits.posZ, data_hits.posY, s=10)
    ax.set_xlabel("X cm")
    ax.set_ylabel("Z cm")
    ax.set_zlabel("Y cm")
    plt.savefig("plot1.jpg")
    plt.show()
    
def plot_cluster_size_hist(hits, title=""):
    num_plots = int(hits["layerID"].max()) + 1
    # if "noise" in [i.lower() for i in title.split(" ")]:
    #     eunit = "keV"
    #     efac = 10e3
    if not num_plots:
        return
    else:
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        for i in range(num_plots):
            # print(pd_data[pd_data.layerID == i])
            data = np.array([utils.edep2csize(utils.edep2medep(edep))\
                for edep in hits[hits.layerID == i]["edep"]])
            if data.size:
                axs[int(i/3), i%3].hist(data, bins=range(0, 16, 1))
                axs[int(i/3), i%3].set_xlim([0, data.max() + (data.max() - data.min())/10])
                axs[int(i/3), i%3].set_xlabel(f"Cluster size")
                axs[int(i/3), i%3].set_ylabel("Entries")
                axs[int(i/3), i%3].set_xlim([0, 20])
        fig.suptitle(f"{title}")
    plt.savefig("./imgs/" + title.replace(" ", "_") + "Clusters.png")

def plot_tcluster_size_hist(hits, title=""):
    num_plots = int(hits["layerID"].max()) + 1
    # if "noise" in [i.lower() for i in title.split(" ")]:
    #     eunit = "keV"
    #     efac = 10e3
    if not num_plots:
        return
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        data = []
        for i in range(num_plots):
            # print(pd_data[pd_data.layerID == i])
            data += [utils.edep2csize(utils.edep2medep(edep))\
                for edep in hits[hits.layerID == i]["edep"]]
        data = np.array(data)
        ax.hist(data, bins=range(0, 16, 1))
        ax.set_xlabel(f"Cluster size")
        ax.set_ylabel("Entries")
        ax.set_xlim([0, 20])
        fig.suptitle(f"{title}")
    plt.savefig("./imgs/" + title.replace(" ", "_") + "Total_Clusters.png")

def plot_edep_hist(hits, title=""):
    num_plots = int(hits["layerID"].max()) + 1
    eunit = "MeV"
    # if "noise" in [i.lower() for i in title.split(" ")]:
    #     eunit = "keV"
    #     efac = 10e3
    if not num_plots:
        return
    else:
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        for i in range(num_plots):
            data = np.array(hits[hits.layerID == i]["edep"])
            if data.size:
                axs[int(i/3), i%3].hist(data, bins=50)
                axs[int(i/3), i%3].set_xlim([0, data.max() + (data.max() - data.min())/10])
                axs[int(i/3), i%3].set_xlabel(f"edep ({eunit})")
                axs[int(i/3), i%3].set_ylabel("Entries")
        fig.suptitle(f"{title}")
    plt.savefig("./imgs/" + title.replace(" ", "_") + ".png")

def plot_bg_hist(data):
    for i in range(len(data)):
        datai = np.array(data[i])
        print(len(data[i]))
        # print(np.histogram(data[i]))
        plt.figure()
        plt.hist(data[i], int(1024*512/256), density=False)
        plt.show()

def plot_HoverC(data, clusters):
    data = np.where(data > 0)
    num_col = 1024
    num_row = 512
    x_list, y_list = [], []
    for c in clusters:
        for p in c:
            x_list.append(int(p%1024))
            y_list.append(int(p/1024))
    hc_data = cluster.get_HfromC(clusters)
    print(np.shape(hc_data))
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.hist2d(data[1], data[0], bins=(range(0, 1024, 1), range(0, 512, 1)), cmap='Greys')
    ax.scatter(x_list, y_list, alpha=0.5, c='green', s=100)
    ax.scatter(hc_data[:, 0], hc_data[:, 1], alpha=0.5, c='red', s=100)
    plt.show()
    
def plot_HoverC_zoom(data, clusters):
    data = np.where(data > 0)
    num_col = 1024
    num_row = 512
    x_list, y_list = [], []
    for c in clusters:
        for p in c:
            x_list.append(int(p%1024))
            y_list.append(int(p/1024))
    hc_data = cluster.get_HfromC(clusters)
    # print(np.shape(hc_data))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist2d(data[1], data[0], bins=(range(0, 1024, 1), range(0, 512, 1)), cmap='Greys')
    ax.scatter(x_list, y_list, alpha=0.5, c='green')
    ax.scatter(hc_data[:, 0], hc_data[:, 1], alpha=0.5, c='red')
    ax.set_xlim([450, 550])
    ax.set_ylim([150, 200])
    x_range = range(450, 570, 20)
    y_range = range(150, 220, 10)
    ax.set_title("Cluster shapes", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_xlabel("X-position of pixel", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_ylabel("Y-position of pixel", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_xticklabels(x_range, 
                                        fontproperties=TIMES_BOLD, 
                                        fontsize=24)
    ax.set_yticklabels(y_range, 
                                        fontproperties=TIMES_BOLD, 
                                        fontsize=24)
    ax.tick_params(axis='both', which='major', 
                                width=2.5, length=10)
    plt.savefig("./imgs/cluster_zoom.png", dpi=300, bbox_inches='tight')
    # plt.show()

def plot_nclusters(data, title=" number of clusters on planes"):
    data = [len(d) for d in data]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(range(len(data)), data, linewidth=5)
    ax.grid(visible=True)
    ax.set_title(title)
    plt.savefig(f"./implot/{title}.png")

def plot_nhit(data, title=" number of hits on planes"):
    data = [len(d) for d in data]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(range(len(data)), data, linewidth=5)
    ax.grid(visible=True)
    ax.set_title(title)
    plt.savefig(f"./implot/{title}.png")

def plot_mon_tracks(mon_data, title=""):
    event_ids = np.sort(np.unique(mon_data["eventID"].values), axis=None).astype(int)
    # print(event_ids)
    tracks = []
    for i in event_ids:
        track_ids = np.unique(mon_data[mon_data.eventID == i]["trackID"].values)
        for j in track_ids:
            tracks.append(mon_data[(mon_data.trackID == int(j)) & (mon_data.eventID == int(i))]\
                .sort_values(by=["layerID"]))
    fig = plt.figure(figsize=(20, 15))
    ax = Axes3D(fig)
    for track in tracks:
        # print(f"{track['eventID'].values}, {track['posZ'].values}")
        ax.plot3D(track["posX"].values, track["posZ"].values, track["posY"].values,
                linewidth=2, c='black')
    ax.view_init(elev=20, azim=-10)
    # view_init_list = [[0, 0], [0, -30]]
    ax.set_xlabel("X mm")
    ax.set_ylabel("Z mm")
    ax.set_zlabel("Y mm")
    ax.set_xlim([0, 30])
    ax.set_ylim([-25, 130])
    ax.set_zlim([0, 13.8])
    plt.savefig(f"./implot/mon_tracks_{title[:-5]}.png")
    
def plot_mon_track(mon_data):
    event_ids = np.sort(np.unique(mon_data["eventID"].values), axis=None).astype(int)
    # print(event_ids)
    tracks = []
    for i in event_ids:
        track_ids = np.unique(mon_data[mon_data.eventID == i]["trackID"].values)
        for j in track_ids:
            tracks.append(mon_data[(mon_data.trackID == int(j)) & (mon_data.eventID == int(i))]\
                .sort_values(by=["layerID"])) 
    for track in tracks:
        # print(track["eventID"])
        fig = plt.figure(figsize=(20, 15))
        ax = Axes3D(fig)
        # print(f"{track['eventID'].values}, {track['posZ'].values}")
        ax.plot3D(track["posX"].values, track["posZ"].values, track["posY"].values,
                linewidth=2, c='black')
        ax.view_init(elev=20, azim=-10)
        # view_init_list = [[0, 0], [0, -30]]
        ax.set_xlabel("X mm")
        ax.set_ylabel("Z mm")
        ax.set_zlabel("Y mm")
        ax.set_xlim([-16, 16])
        ax.set_ylim([-25, 130])
        ax.set_zlim([-7.5, 7.5])
        plt.savefig(f"./implot/mon-tracks/event_{(track['eventID'].values)[0]}_\
track_{(track['trackID'].values)[0]}.png")

def plot_colmon_track(colmon_tracks):
    fig = plt.figure(figsize=(20, 15))
    ax = Axes3D(fig)
    for track in colmon_tracks:
        ax.plot3D(track["posX"].values, track["posZ"].values, track["posY"].values,
                linewidth=4, c='black')
        ax.scatter(track["posX"].values, track["posZ"].values, track["posY"].values,
                s=100, c='red')
        ax.view_init(elev=10, azim=-10)
        ax.set_xlabel("X mm")
        ax.set_ylabel("Z mm")
        ax.set_zlabel("Y mm")
        ax.set_xlim([-16, 16])
        ax.set_ylim([-25, 130])
        ax.set_zlim([-7.5, 7.5])
    plt.savefig(f"./implot/mon-col_tracks.png")
        
def plot_rec_tracks(data, scatter=False, name=' ', kind="Experiment", inpixel=False):
    labels = ["x mm", "y mm", "z mm"]
    facs = [1024/30, 512/13.8]
    lims = [[0, 30], [0, 15], [25, 181]]
    fig = plt.figure(figsize=(20, 10))
    ax = Axes3D(fig)
    data_plot = [] 
    # if kind.lower() == 'simulation':
    #     if inpixel:
    #         track_inpix = []
    #         ax.plot3D(data_plot["posX"].values, data_plot["posZ"].values)
    #     if scatter:
    #         for track in data:
    #             ax.scatter([j.data["posX"]*facs[0] for j in track], 
    #                     [(j.data["posZ"] - 50)/25 for j in track], 
    #                     [j.data["posY"]*facs[1] for j in track],
    #                     c='red', s=100)
    # else:
    #     for track in data:
    #         ax.plot3D([j.data["posX"] for j in track], [j.data["posZ"] for j in track], [j.data["posY"] for j in track],
    #                 linewidth=4, c='black')
    #     if scatter:
    #         for track in data:
    #             ax.scatter([j.data["posX"] for j in track], [j.data["posZ"] for j in track], [j.data["posY"] for j in track],
    #                     c='red', s=100)
    # else:
    for evt_i, evt in  enumerate(np.unique(data["eventID"].values)):
        for track_i, track in enumerate(np.unique(data[data.eventID == evt]["MyTrackID"].values)):
            data_pos = data[(data.eventID == evt) & (data.MyTrackID == track)]
            if inpixel:
                data_pos["posX"] = data_pos["posX"].values*facs[0]
                data_pos["posY"] = data_pos["posY"].values*facs[1]
                data_pos["posZ"] = data_pos["posZ"].values/25
            data_plot.append(data_pos)
    for i, d in enumerate(data_plot):
        ax.plot3D(data_plot[i]["posX"].values, data_plot[i]["posZ"].values,\
            data_plot[i]["posY"].values, linewidth=3, c='k')
        if scatter:
            ax.scatter(data_plot[i]["posX"].values, data_plot[i]["posZ"].values,\
                data_plot[i]["posY"].values, c='red', s=100)
    if inpixel:
        labels = ["x-pixel", "y-pixel", "Sensor layer"]
        lims = [[0, 1024], [0, 512], [0, 5]]
                 
    ax.view_init(elev=10, azim=-10)
    # view_init_list = [[0, 0], [0, -30]]
    ax.set_xlabel(f"{labels[0]}", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE, labelpad=30)
    ax.set_ylabel(f"{labels[2]}", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE, labelpad=30)
    ax.set_zlabel(f"{labels[1]}", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE, labelpad=30)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[2])
    ax.set_zlim(lims[1])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    for z in ax.zaxis.get_major_ticks():
        z.label.set_fontproperties(TIMES_BOLD)
        z.label.set_fontsize(FONT_SIZE)
    ax.grid(False)
    xx = np.linspace(0, 1024, 30)
    yy = np.linspace(0, 512, 15)
    X, Y = np.meshgrid(xx, yy)
    for i in range(6):
        Z = np.ones(np.shape(xx))*(i)
        ax.plot_surface(X, Z, Y, alpha=0.2)
    # ax.annotate(f"70 MeV", 
    #             xy=(800,1000), xycoords='axes pixels',
    #             size=FONT_SIZE, ha='center', va='top', fontproperties=TIMES_BOLD)
    # plt.subplots_adjust(wspace=0, hspace=0)
    ax.set_box_aspect([1, 2, 1])  # [width, height, depth]
    plt.savefig(f"./output/imgs/reconstruction/tracks/{kind}_{name.replace(' ', '_')}_rec_tracks.png",
                bbox_inches='tight')
        
def plot_cmp_clusters(data):
    labels = [f"PMMA {i} cm" for i in [3.9, 4.0, 4.1]] + ["Cu collimator"]
    fig, axs = plt.subplots(4, figsize=(20, 20), sharex=True)
    y_range = [
        range(0, 801, 100),
        range(0, 701, 100),
        range(0, 51, 10),
        range(44, 55, 2)
    ]
    # print(np.array(data))
    colors = ['orange']*3 + ['green']
    for i in range(len(data)):
        data_y = [len(l_data) for l_data in data[i]]
        data_y = [data_y[0]] + data_y + [data_y[-1]]
        dy = max(data_y) - min(data_y)
        axs[i].step(range(-1, len(data[i]) + 1), data_y,\
            where='mid', linewidth=5, label=labels[i], c=colors[i])
        axs[i].set_xlim([0, 5])
        axs[i].set_ylim([min(data_y) - dy/10, max(data_y) + dy/4])
        axs[i].annotate(labels[i], xy=(1650, 410), xycoords='axes pixels',
                size=28, ha='right', va='top', fontproperties=TIMES_BOLD)
        axs[i].tick_params(axis='both', which='major', width=2.5, length=8)
        axs[i].set_xticks(range(6))
        axs[i].set_yticks(y_range[i])
        axs[i].set_xticklabels(range(6), 
                                           fontproperties=TIMES_BOLD, 
                                           fontsize=24)
        axs[i].set_yticklabels(y_range[i], 
                                           fontproperties=TIMES_BOLD, 
                                           fontsize=24)
        # if i == 0:
        #     axs[i].set_ylabel("Entries\n\n")
        # if i == len(data) - 1:
        #     axs[i].set_xlabel("\n\nLayer"
    fig.supylabel("Entries\n\n", fontproperties=TIMES_BOLD, fontsize=32)
    fig.supxlabel("\n\nLayer", fontproperties=TIMES_BOLD, fontsize=32)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("./imgs/cmp_nclusters.png")

def plot_cluster_4hist(data):
    labels = [f"PMMA {i} cm" for i in [3.9, 4.0, 4.1]] + ["Cu collimator"]
    colors = ['orange']*3 + ['green']
    x_ranges = [
        range(0, 81, 10),
        range(0, 81, 10),
        range(0, 36, 5),
        range(0, 101, 10)
    ]
    y_ranges = [
        range(0, 161, 20), 
        range(0, 161, 20),
        range(0, 10, 1),
        range(0, 81, 10)
    ]
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    for i in range(len(data)):
        data_h = []
        for d in data[i]:
            data_h = data_h + [len(c) for c in d]
        x_max = int(max(data_h) + max(data_h)/5)
        axs[int(i/2), i%2].hist(data_h, bins=range(0, x_max), color=colors[i])
        axs[int(i/2), i%2].set_ylabel(f"Entries", loc='top', fontproperties=TIMES_BOLD,
                                      fontsize=26)
        axs[int(i/2), i%2].set_xlabel(f"Cluster size", loc='right', fontproperties=TIMES_BOLD,
                                      fontsize=26) 
        axs[int(i/2), i%2].set_xticks(x_ranges[i])
        axs[int(i/2), i%2].set_yticks(y_ranges[i])
        axs[int(i/2), i%2].set_xticklabels(x_ranges[i], 
                                           fontproperties=TIMES_BOLD, 
                                           fontsize=24)
        axs[int(i/2), i%2].set_yticklabels(y_ranges[i], 
                                           fontproperties=TIMES_BOLD, 
                                           fontsize=24)
        axs[int(i/2), i%2].xaxis.set_minor_locator(AutoMinorLocator())
        axs[int(i/2), i%2].yaxis.set_minor_locator(AutoMinorLocator())
        axs[int(i/2), i%2].tick_params(axis='both', which='major', 
                                                direction="in",
                                                width=2.5, length=10)
        axs[int(i/2), i%2].tick_params(axis='both', which='minor', 
                                                direction="in",
                                                width=1.5, length=6)
        axs[int(i/2), i%2].annotate(labels[i], xy=(680,680), xycoords='axes pixels',
                size=28, ha='right', va='top', fontproperties=TIMES_BOLD)
    # fig.supylabel("Entries\n\n", fontproperties=TIMES_BOLD, fontsize=32)
    # fig.supxlabel("\n\nCluster size", fontproperties=TIMES_BOLD, fontsize=32)
    plt.tight_layout()
    plt.savefig("./imgs/cluster4.png", dpi=300)

def plot_cluster_hist(data, name="", title=""):
    fig, ax = plt.subplots(figsize=(10, 8))
    data_h = []
    for d in data:
        data_h = data_h + [len(c) for c in d if len(c)]
    ax.hist(data_h, bins=range(0, int(max(data_h)) + 1), color='orange')
    ax.set_ylabel(f"Entries", loc='top', fontproperties=TIMES_BOLD,
                                    fontsize=24)
    ax.set_xlabel(f"Cluster size", loc='right', fontproperties=TIMES_BOLD,
                                    fontsize=24) 
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', 
                                            direction="in",
                                            width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', 
                                            direction="in",
                                            width=1.5, length=6)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(24)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(24)
    ax.set_title(f"{title}", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_xlim([0, 40])
    ax.yaxis.get_major_ticks()[0].label.set_visible(False)
    plt.savefig(f"./imgs/{name}.png", dpi=300 ,bbox_inches='tight')
    # plt.show()

def plot_csize_hist(data, name="", title=""):
    data = [int(d) for d in data]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(data, bins=range(0, int(max(data)) + 1), color='orange')
    ax.set_ylabel(f"Entries", loc='top', fontproperties=TIMES_BOLD,
                                    fontsize=24)
    ax.set_xlabel(f"Cluster size", loc='right', fontproperties=TIMES_BOLD,
                                    fontsize=24) 
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', 
                                            direction="in",
                                            width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', 
                                            direction="in",
                                            width=1.5, length=6)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(24)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(24)
    ax.set_title(f"{title}", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_xlim([0, 40])
    ax.yaxis.get_major_ticks()[0].label.set_visible(False)
    plt.savefig(f"./imgs/{name}.png", dpi=300 ,bbox_inches='tight')

def plot_cluster_hist_sel(data, sel=[]):
    data = data[data.layerID == 0]
    fig, ax = plt.subplots(figsize=(10, 8))
    x_data = data["posX"].values
    y_data = data["posY"].values
    edep_data = data["edep"].values
    edep_sel = edep_data[((x_data - sel["x"])**2)/sel["a"]**2 + ((y_data - sel["y"])**2)/sel["b"]**2 <= 1]
    cluster_size_sel = [utils.edep2csize(utils.edep2medep(e)) for e in edep_sel]
    ax.hist(cluster_size_sel, bins=range(0, int(cluster_size_sel)) + 1, color='orange')
    ax.set_ylabel(f"Entries", loc='top', fontproperties=TIMES_BOLD,
                                    fontsize=24)
    ax.set_xlabel(f"Cluster size", loc='right', fontproperties=TIMES_BOLD,
                                    fontsize=24) 
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', 
                                            direction="in",
                                            width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', 
                                            direction="in",
                                            width=1.5, length=6)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(24)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(24)
    ax.set_title("Cluster size distribution", fontproperties=TIMES_BOLD, fontsize=24)
    plt.savefig("./imgs/cluster_sim_sel.png", dpi=300 ,bbox_inches='tight')

def plot_cluster_hist_colmon(data, name="", title=""):
    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_size = [utils.edep2csize(utils.edep2medep(e)) for e in data]
    ax.hist(cluster_size, bins=range(0, int(max(cluster_size)) + 1), align='mid',color='gray')
    ax.set_ylabel(f"Entries", loc='top', fontproperties=TIMES_BOLD,
                                    fontsize=24)
    ax.set_xlabel(f"Cluster size", loc='right', fontproperties=TIMES_BOLD,
                                    fontsize=24) 
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', 
                                            direction="in",
                                            width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', 
                                            direction="in",
                                            width=1.5, length=6)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(24)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(24)
    ax.set_title(f"{title}", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_xlim([0, 40])
    ax.yaxis.get_major_ticks()[0].label.set_visible(False)
    plt.savefig(f"./imgs/{name}.png", dpi=300 ,bbox_inches='tight')    

def plot_HoverH(data, hover=False, mul=True,zoom=[], name="", title="", elps=[]):
    hsv_mod = cm.get_cmap('hsv', 256)
    color_list = hsv_mod(np.linspace(0, 0.7, 16)).tolist()
    # print(color_list) 
    color_list.append([1, 1, 1, 1])
    color_list.reverse()
    new_cmap = ListedColormap(color_list) if mul else 'Blues'
    num_col = 1025
    num_row = 513
    
    # num_col = 100
    # num_row = 50
    fig, ax = plt.subplots(figsize=(16, 8))
    h = ax.hist2d(data[:, 0], data[:, 1], bins=(range(0, num_col, 1), 
                                            range(0, num_row, 1)), cmap=new_cmap)
    if hover:
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, c='red', s=10)
    if zoom:
        ax.set_xlim(zoom[0])
        ax.set_ylim(zoom[1])
    else: 
        ax.set_xlim([0, 1024])
        ax.set_ylim([0, 512])
        
    ax.tick_params(axis='both', which='major', 
                                width=2.5, length=10)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(24)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(24)
    if mul:
        cbar = plt.colorbar(h[3], ax=ax, pad=0.008)
        for t in cbar.ax.get_yticklabels():
            t.set_fontproperties(TIMES_BOLD)
            t.set_fontsize(24)
    ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_xlabel("X-position of pixel", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_ylabel("Y-position of pixel", fontproperties=TIMES_BOLD, fontsize=24)

    if elps:
        elipse = Ellipse(xy=(elps["x"], elps["y"]), width=elps["a"]*2, height=elps["b"]*2,
                         edgecolor='k', fc=None, lw=2, fill=False)
        ax.add_patch(elipse)
    # plt.savefig(f"./imgs/{name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_beam1D(data, prms):
    plt.figure(figsize=(16,12))
    plt.scatter(data[0], data[1])
    plt.plot(data[0], [Gauss(i, *prms) for i in data[0]], c='red')
    plt.xlabel("d")
    plt.ylabel("Entries")
    plt.show()

def plot_beam_dist(data, mean_sigma_data, name="", pdf=True, cmin=False, bi=False,
                   lim_fits={"x": [0, 1024], "y": [0, 512]}, subt2=False, subt1=False):
    mean_sigma_data2 = []
    for i in range(len(mean_sigma_data["layerID"])):
        # print((mean_sigma_data[mean_sigma_data.layerID == i]["paramsX"].values[0])[0])
        mean_sigma_data2.append([mean_sigma_data[mean_sigma_data.layerID == i]["paramsX"].values[0], 
                                 mean_sigma_data[mean_sigma_data.layerID == i]["paramsY"].values[0]])
    for i in range(2):
        fig, axs = plt.subplots(2, 3, figsize=(16, 12))
        for j in range(len(mean_sigma_data2)):
            data_plot = data[data.layerID == j][f"pos{['X', 'Y'][i]}"].values
            data_plot = data_plot[np.logical_and(data_plot > lim_fits[f"{['x', 'y'][i]}"][j][0], 
                                                 data_plot < lim_fits[f"{['x', 'y'][i]}"][j][1])]
            if cmin:
                values, counts = np.unique(data_plot, return_counts=True)
                c_min = counts.min()
                counts = counts - c_min
                data_plot = np.repeat(values, counts)
            if subt2:
                values, counts = np.unique(data_plot, return_counts=True)
                counts = counts if cmin else counts - counts.min()
                data_subt2 = gauss(values, *(mean_sigma_data2[j][i][3:]))
                counts2 = (counts - data_subt2).astype(np.int64)
                counts2[counts2 < 0] = 0
                data_plot = np.repeat(values, counts2)
            if subt1:
                values, counts = np.unique(data_plot, return_counts=True)
                counts = counts if cmin else counts - counts.min()
                data_subt1 = gauss(values, *(mean_sigma_data2[j][i][:3]))
                counts2 = (counts - data_subt1).astype(np.int64)
                counts2[counts2 < 0] = 0
                data_plot = np.repeat(values, counts2)
            n, bins, patches = axs[int(j/3), j%3].hist(data_plot, bins=range(int(data_plot.min()), 
                                                                    int(data_plot.max())), 
                                        density=pdf, facecolor='green', alpha=0.75)
            # print(f"{len(n)}, {len(bins)}\n")
            if not bi:
                axs[int(j/3), j%3].plot(bins, gauss(bins, *(mean_sigma_data2[j][i])), 'r--', linewidth=4)
                axs[int(j/3), j%3].annotate(f"$\mu$ = {mean_sigma_data2[j][i][0]: .0f}\n$\sigma$ = {mean_sigma_data2[j][i][1]: .2f}", 
                                            xy=(900,1200), xycoords='axes pixels',
                            size=FONT_SIZE, ha='center', va='top', fontproperties=TIMES_BOLD)
            else:
                if subt2:
                    axs[int(j/3), j%3].plot(bins, gauss(bins, *(mean_sigma_data2[j][i][:3])), 'r--', linewidth=4)
                    axs[int(j/3), j%3].annotate(f"$\mu1$ = {mean_sigma_data2[j][i][0]: .0f}\n\
$\sigma1$ = {mean_sigma_data2[j][i][1]: .2f}\n", xy=(900,1200), xycoords='axes pixels',
                            size=FONT_SIZE, ha='center', va='top', fontproperties=TIMES_BOLD)
                else:
                    if not subt1:
                        axs[int(j/3), j%3].plot(bins, gauss(bins, *(mean_sigma_data2[j][i][:3])), 'r--', linewidth=4)
                    if not subt2:
                        axs[int(j/3), j%3].plot(bins, gauss(bins, *(mean_sigma_data2[j][i][3:])), 'b--', linewidth=4)
                    axs[int(j/3), j%3].annotate(f"$\mu1$ = {mean_sigma_data2[j][i][0]: .0f}\n\
$\sigma1$ = {mean_sigma_data2[j][i][1]: .2f}\n\
$\mu2$ = {mean_sigma_data2[j][i][3]: .0f}\n\
$\sigma2$ = {mean_sigma_data2[j][i][4]: .2f}", xy=(900,1200), xycoords='axes pixels',
                            size=FONT_SIZE, ha='center', va='top', fontproperties=TIMES_BOLD)
            # axs[int(j/3), j%3].grid(linestyle = ':', linewidth = 2)
            axs[int(j/3), j%3].set_title(f"layer {j}", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
            axs[int(j/3), j%3].set_xlabel(f"pixel", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
            axs[int(j/3), j%3].set_ylabel("Entries", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
            for x in axs[int(j/3), j%3].xaxis.get_major_ticks():
                x.label.set_fontproperties(TIMES_BOLD)
                x.label.set_fontsize(FONT_SIZE)
            for y in axs[int(j/3), j%3].yaxis.get_major_ticks():
                y.label.set_fontproperties(TIMES_BOLD)
                y.label.set_fontsize(FONT_SIZE)
        plt.savefig(f"./output/imgs/beam/{name}_{['X', 'Y'][i]}.png", dpi=300, bbox_inches='tight')
        # plt.show()

def plot_beam_sigma(data, name="", title="", pct_fit=False): 
    x_range = np.array(range(6)) * 25 + 50 #mm
    s = ['o', '^', 's', 'x', 'p', '^']
    cs = ['b', 'r']*3
    ens = [str(e) + " MeV" for e in [70, 100, 120, 150, 180, 200]]
    fig, ax = plt.subplots(figsize=(14, 10))
    data_y = [[] for i in range(len(data.keys()))]
    for i in range(6):
        for j in range(len(data.keys())):
            data_y[j].append((data[ens[j]]["sigmas"]["x"][i] +\
                data[ens[j]]["sigmas"]["y"][i])/2)
    # for i in range(len(data_y[0])):
    #     print("{", end=' ')
    #     print(",".join([f"{j*0.28: .2f} *mm" for j in np.array(data_y)[i, :]]), end=' ')
    #     print("},")
            # data_y[1].append((data[1]["paramsX"].values[i][1] + data[1]["paramsY"].values[i][1])/2)
    for i in [0, len(data_y) - 1]:
        ax.scatter(np.arange(6), np.array(data_y[i]), marker=s[i], s=120, zorder = 15,
                   label=ens[i], c='k')
        popt, pcov = np.polyfit(np.arange(6)*25, np.log(data_y[i]), 1, w=np.sqrt(data_y[i]), cov=True)
        a = np.exp(popt[1])
        b = popt[0]
        ax.plot(np.arange(6), np.array(data_y[i]), zorder = 10,
                            linewidth=3, c=cs[i])
        # ax.plot(np.arange(-1, 6)*25, np.array(a*np.exp(b*np.arange(-1, 6)*25)), zorder = 10, linestyle='--',
        #                     linewidth=3, c=cs[i])
        # print(np.array(data_y[i])*0.028)
        # print(f" {0.028*a*np.exp(b*(-1)*25)} *mm,")
    
    if pct_fit:
        pct_data = pd.read_csv("./data/simulation/beamfit0.5d.csv", index_col=None)
        ax.scatter(range(6), pct_data['sigma']/0.028, label='pCT fit')
        ax.plot(np.arange(6), pct_data['sigma']/0.028, zorder = 10,
                            linewidth=2, c=cs[i], linestyle='--', alpha=0.5)
        # print("pass")
    # ax.grid(linestyle = ':', linewidth = 2)
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    ax.set_xlabel("Sensor layer", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("Spot sigma (pixels)", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(title,
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    plt.savefig(f"./output/imgs/beam/{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_contour_cmpl(cmpl, name="", title="", nhit=2):
    fig, ax = plt.subplots(figsize=(14, 10))
    xlist = [*range(1, nhit+1)]
    ylist = np.linspace(0.2, 10, 50)*20
    data = []

    for key, value in cmpl.items():
        data.append(value)
    X, Y = np.meshgrid(xlist, ylist)
    cp = ax.contourf(X, Y, np.array(data).transpose(), level=14, cmap = 'gnuplot')
    # norm= Normalize(vmin=cp.cvalues.min(), vmax=cp.cvalues.max())
    # sm = plt.cm.ScalarMappable(norm=norm, cmap = 'gnuplot')
    # sm.set_array([])
    # fig.colorbar(sm, ticks=cs.levels)
    # fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title(name, fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_xlabel('Number of hits', fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel('Smax (mrad)', fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylim([min(ylist), 150]) 
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    cbar = plt.colorbar(cp, ax=ax, pad=0.008)
    for t in cbar.ax.get_yticklabels():
        t.set_fontproperties(TIMES_BOLD)
        t.set_fontsize(24)
    plt.show()

def plot_eff_smax(eff_dict, title="", name=""):
    x_range = np.linspace(0.2, 10, 50)*20
    fig, ax = plt.subplots(figsize=(14, 10))
    for key, value in eff_dict.items():
        ax.plot(x_range, np.array(value) * 100, linewidth=3, 
                label=f"{key}")
        # print("pass")
    ax.grid(linestyle = ':', linewidth = 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    ax.set_xlabel("Smax (mrad)", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("eff (%)", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    ax.set_xlim([x_range[0], 150])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=24)
    plt.savefig(f"./imgs/eff_smax{'_' if name else ''}{name}.png", bbox_inches='tight', dpi=300)
    plt.show()

def plot_nchit(chit_path):
    data = np.genfromtxt(chit_path)
    print(data)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(data[:, 1], bins=range(0, int(max(data[:, 1]))))
    plt.show()

def plot_beam2d(data, lim=[[0, 1024], [0, 512]], name="", title="", bi=False):
    size = 100
    x = np.linspace(lim[0][0], lim[0][1], size)
    y = np.linspace(lim[1][0], lim[1][1], size)
    X, Y = np.meshgrid(x, y)
    x_ = X.flatten()
    y_ = Y.flatten()
    xy = np.vstack((x_, y_)).T
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))
    for j in range(6):
        data2 = data[data.layerID == j]
        # if not bi:
        mus = [utils.pdlist2list(data2["paramsX"].values[0])[0], 
                utils.pdlist2list(data2["paramsY"].values[0])[0]]
        sigmas = [utils.pdlist2list(data2["paramsX"].values[0])[1], 
                utils.pdlist2list(data2["paramsY"].values[0])[1]]
        covs = [k**2 for k in sigmas]
        normal_rv = multivariate_normal(mus, cov=[[covs[0], 0], [0, covs[1]]])
        z = normal_rv.pdf(xy)
        Z = z.reshape(size, size, order='F')
        axs[int(j/3), j%3].contourf(X, Y, Z.T)
        axs[int(j/3), j%3].xaxis.set_minor_locator(AutoMinorLocator())
        axs[int(j/3), j%3].yaxis.set_minor_locator(AutoMinorLocator())
        axs[int(j/3), j%3].tick_params(which='major', width=1.5, length=18, direction="in")
        axs[int(j/3), j%3].tick_params(which='minor', width=1, length=8, direction="in")
        axs[int(j/3), j%3].set_xlabel("Distance from isocenter (cm)", loc="right", 
                    fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        axs[int(j/3), j%3].set_ylabel("Spot sigma (mm)", loc="top",
                    fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        axs[int(j/3), j%3].set_title(title,
                    fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        # leg = axs[int(j/3), j%3].legend(prop={'fname': FNAME, 'size': FONT_SIZE})
        # leg.get_frame().set_edgecolor('black')
        # leg.get_frame().set_boxstyle('Square', pad=0.3)
        for x in axs[int(j/3), j%3].xaxis.get_major_ticks():
            x.label.set_fontproperties(TIMES_BOLD)
            x.label.set_fontsize(FONT_SIZE)
        for y in axs[int(j/3), j%3].yaxis.get_major_ticks():
            y.label.set_fontproperties(TIMES_BOLD)
            y.label.set_fontsize(FONT_SIZE)
    plt.show()
    
def plot_exp_nclusters(pd_data, name="", title="", filts=[]): 
    fig, ax = plt.subplots(figsize=(14, 10))
    data_70 = []
    data_200 = []
    for evt in np.sort(np.unique(pd_data[pd_data.energy == 70]['eventID'].values)):
        data_70.append(np.sum(pd_data[(pd_data.energy == 70) &\
            (pd_data.eventID == evt)]['clusterSize'].values))
    for evt in np.sort(np.unique(pd_data[pd_data.energy == 200]['eventID'].values)):
        data_200.append(np.sum(pd_data[(pd_data.energy == 200) &\
            (pd_data.eventID == evt)]['clusterSize'].values))
    print(f"70 MeV: {np.mean(data_70)}, 200 MeV: {np.mean(data_200)}")
    ax.plot(range(800), data_70[:800], label="70 MeV", linewidth=2, c='b')
    ax.plot(range(800), data_200[:800], linestyle='--', label="200 MeV", linewidth=2, alpha=0.6, c='r')
    # ax.grid(linestyle = ':', linewidth = 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    ax.set_xlabel("Event number", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("Counts", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    ax.set_xlim([0, 800])
    # ax.set_ylim([0, 300])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=24)
    plt.savefig(f"./output/imgs/cluster/ncluster_event_{name}.png", bbox_inches='tight', dpi=300)
    plt.show()

def plot_2D_beamsigma(pd_data, name="", title="", energy=70, layer=0): 
    hsv_mod = cm.get_cmap('hsv', 256)
    color_list = hsv_mod(np.linspace(0, 0.7, 16)).tolist()
    # print(color_list) 
    color_list.append([1, 1, 1, 1])
    color_list.reverse()
    new_cmap = ListedColormap(color_list)
    fig, ax = plt.subplots(figsize=(14, 10))
    data = pd_data[(pd_data.energy == energy) & (pd_data.layerID == layer)]
    ax.hist2d(data["posX"].values, data["posY"].values, bins=(range(0, data["posX"].values.max(), 1), 
                                                range(0, data["posY"].values.max(), 1)), cmap=new_cmap)
    plt.show()

def plot_HoverH2(data, hover=False, zoom=[], name="", title=""):
    hsv_mod = cm.get_cmap('jet', 300)
    num_col = 1024
    num_row = 512
    
    # valuesx, countsx = np.unique(data["posX"].values)
    # valuesy, countsy = np.unique(data["posX"].values)
    # count_max = countsx.max() if countsx.max() > countsy.max() else countsy.max()
    color_list = hsv_mod(np.linspace(0, 1.0, 30)).tolist()
    new_cmap = ListedColormap(color_list)
    fig, ax = plt.subplots(figsize=(12, 10))
    h = ax.hist2d(data["posX"].values, data["posY"].values, bins=(range(0, num_col, 1), 
                                            range(0, num_row, 1)), cmap=new_cmap)
    if hover:
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, c='red', s=10)
    if zoom:
        ax.set_xlim(zoom[0])
        ax.set_ylim(zoom[1])
    else: 
        ax.set_xlim([0, 1024])
        ax.set_ylim([0, 512])
        
    ax.tick_params(axis='both', which='major', 
                                width=2.5, length=10)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    cbar = plt.colorbar(h[3], ax=ax, pad=0.008)
    if title in ["70 MeV", "100 MeV"]:
        cbar.ax.locator_params(nbins=5)
        # cbar = plt.colorbar(h[])
    for t in cbar.ax.get_yticklabels():
        t.set_fontproperties(TIMES_BOLD)
        t.set_fontsize(FONT_SIZE)
    ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_xlabel("X-position of pixel", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("Y-position of pixel", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)

    plt.savefig(f"./imgs/Beam/{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_2sigma_entries(data_hits, data_beam, name="", title=""): 
    x_range = np.array(range(6)) * 25 + 50 #mm
    ens = [str(e) + " MeV" for e in [70, 100, 120, 150, 180, 200]]
    fig, ax = plt.subplots(figsize=(14, 10))
    for e in np.unique(data_beam["energy"].values):
        data_2sigs = [[] for i in range(6)]
        for l in data_beam[data_beam.energy == e]["layerID"].values:
            for evt in data_hits[e]["eventID"].values:
                meanX = data_beam[(data_beam.energy == e) & (data_beam.layerID == l)]["meanX"].values[0]
                meanY = data_beam[(data_beam.energy == e) & (data_beam.layerID == l)]["meanY"].values[0]
                sigmaX = data_beam[(data_beam.energy == e) & (data_beam.layerID == l)]["sigmaX"].values[0]
                sigmaY = data_beam[(data_beam.energy == e) & (data_beam.layerID == l)]["sigmaY"].values[0]
                lim = [[meanX - 2*sigmaX, meanX + 2*sigmaX], [meanY - 2*sigmaY, meanY + 2*sigmaY]]
                data_2sigs[l].append(data_hits[e][(data_hits[e].layerID == l) & (data_hits[e].eventID == evt) &
                                            (data_hits[e].posX > lim[0][0]) & 
                                            (data_hits[e].posX < lim[0][1]) &
                                            (data_hits[e].posY > lim[1][0]) &
                                            (data_hits[e].posY < lim[0][1])])
    
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    ax.set_xlabel("Sensor layer", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("Spot sigma (pixels)", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(title,
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    # plt.savefig(f"./output/imgs/beam/{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_hist_cluster(data, kind="exp", name="", title="", sp=''): 
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(data["clusterSize"].values, bins=range(data["clusterSize"].min(), data["clusterSize"].max() + 1, 1),
            label=f"{len(data['clusterSize'].values)}", linewidth=2, color = "lightblue", ec='k', align='left')
    ax.set_xlabel("Cluster size", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_xlim([0, 35])
    ax.set_ylabel("Counts", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(title,
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    # ax.grid(linestyle = ':', linewidth = 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    # leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    # leg.get_frame().set_edgecolor('black')
    # leg.get_frame().set_boxstyle('Square', pad=0.3)
    ax.text(0.99, 0.98, f"Entries          {len(data['clusterSize']): 6d}\n\
Mean            {np.mean(data['clusterSize'].values): .3f}\nStd Dev      \
  {np.std(data['clusterSize'].values): .3f}", bbox=dict(fc='w'),
            transform=plt.gca().transAxes, size=FONT_SIZE, ha='right', va='top', fontproperties=TIMES_BOLD)
    plt.savefig(f"./output/imgs/cluster/sigma_cluster_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plotbox_edep_and_cluster(data, kind="Cluster size", name="", title=""): 
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.boxplot( 
            [v["clusterSize"].values for k, v in data.items()],
            # yerr=[np.std(v["clusterSize"].values) for k, v in data.items()],
            # fmt='o'
            whiskerprops = dict(linewidth=2.0, color='black'),
            boxprops = dict(linewidth=2.0, color='black'),
            medianprops = dict(linewidth=2.0, color='red'),
            capprops = dict(linewidth=2.0, color='black'),
            flierprops= {'marker': 'o', 'markersize': 10, 'markerfacecolor': 'k'}
            )
    ax.set_xticklabels([70, 100, 120, 150, 180, 200])
    ax.set_ylim([0, 35])
    ax.set_xlabel("Energy (MeV)", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel(f"{kind}", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(title,
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_tick_params(which='major', width=1.5, length=18, direction="in")
    ax.yaxis.set_tick_params(which='minor', width=1, length=8, direction="in")
    ax.grid(linestyle = ':', linewidth = 2)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    # leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    # leg.get_frame().set_edgecolor('black')
    # leg.get_frame().set_boxstyle('Square', pad=0.3)
    plt.savefig(f"./output/imgs/cluster/{kind.replace(' ', '_').lower()}_{name}.png", dpi=300, bbox_inches='tight')
    # plt.show()

def plot_avg_edep_and_cluster(data, data_sim, kind="cluster size", name="", title=""): 
    data_plot = []
    for e in np.unique(data["energy"].values):
        data_list = []
        for evt in np.unique(data[data.energy == e]["eventID"].values):
            data_list.append(np.mean(data[(data.energy == e) & (data.eventID == evt)]["clusterSize"].values))
        data_plot.append(data_list)
    fig, ax = plt.subplots(figsize=(14, 10))
    # ax.errorbar([data_sim[f"{i} MeV"]["mean energy"]*1e3 for i in [70, 100, 120, 150, 180, 200]], 
    #         [np.mean(d) for d in data_plot],
    #         yerr=[np.std(d) for d in data_plot],
    #         marker='o', ms=16, ecolor='red', fmt=' ', elinewidth=4)
    x = [data_sim[f"{i} MeV"]["mean energy"]*1e3 for i in [70, 100, 120, 150, 180, 200]]
    y = [np.mean(d) for d in data_plot]
    ax.scatter(x, y, marker='X', s=150, label="Data")
    popt, pcov = curve_fit(lambda x, a, b: a*(x**b), x, y, p0=[4.23, 0.65])
    print(popt)
    ax.plot(x, [popt[0]*(en**popt[1]) for en in x], color='r', alpha=0.7, linewidth=3, label="Curve fitting")
    # ax.plot(x, [])
    # ax.set_xticklabels([70, 100, 120, 150, 180, 200])
    # ax.set_ylim([0, 35])
    ax.set_xlabel("Mean deposited energy (keV)", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel(f"Mean {kind} (pixels)", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(title,
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylim([0, 15])
    ax.set_xlim([12, 30])
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_tick_params(which='major', width=1.5, length=18, direction="in")
    ax.yaxis.set_tick_params(which='minor', width=1, length=8, direction="in")
    ax.xaxis.set_tick_params(which='minor', width=1, length=8, direction="in")
    ax.xaxis.set_tick_params(which='major', width=1.5, length=18, direction="in")
    # ax.grid(linestyle = ':', linewidth = 4)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
        x.set_pad(10)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
        y.set_pad(10)
    # ax.set_xticks([70, 100, 120, 150, 180, 200])
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    # plt.savefig(f"./output/imgs/cluster/mean_{kind.replace(' ', '_').lower()}_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_cluster_count(data, title="", name=""):
    data_plot = []
    for e in np.unique(data["energy"].values):
        data_pplot = []
        for evt in np.unique(data[data.energy == e]["eventID"].values):
            data_evt = data[(data.eventID == evt) & (data.energy == e)]["clusterID"].values
            if data_evt.size:
                data_pplot.append(len(np.unique(data_evt)))
        data_plot.append([np.mean(data_pplot), np.std(data_pplot)])
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.errorbar([70, 100, 120, 150, 180, 200], 
            [data_plot[i][0] for i in range(len(data_plot))],
            yerr=[data_plot[i][1] for i in range(len(data_plot))],
            marker='o', ms=16, ecolor='red', fmt=' ')
    # ax.set_xticklabels([70, 100, 120, 150, 180, 200])
    # ax.set_ylim([0, 35])
    ax.set_xlabel("Mean deposited energy (keV)", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel(f"Number of cluster", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(title,
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_tick_params(which='major', width=1.5, length=18, direction="in")
    ax.yaxis.set_tick_params(which='minor', width=1, length=8, direction="in")
    ax.xaxis.set_tick_params(which='minor', width=1, length=8, direction="in")
    ax.xaxis.set_tick_params(which='major', width=1.5, length=18, direction="in")
    ax.grid(linestyle = ':', linewidth = 4)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    # ax.set_xticks([70, 100, 120, 150, 180, 200])
    # leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    # leg.get_frame().set_edgecolor('black')
    # leg.get_frame().set_boxstyle('Square', pad=0.3)
    plt.savefig(f"./output/imgs/cluster/number_of_cluster_events.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_nhits0(*args, **kwargs):
    data = args[0]
    fig, ax = plt.subplots(figsize=(14, 10))
    values, bins, _ = ax.hist(data[0][:, 1], bins=range(data[0][:, 1].min(), data[0][:, 1].max() + 1, 1),
            histtype='step', fill=False, linewidth=4, label="70 MeV", align='left', color='b')
    print(f"70 MeV: {sum(np.diff(bins)*values)}")
    values, bins, _ = ax.hist(data[1][:, 1], bins=range(data[1][:, 1].min(), data[1][:, 1].max() + 1, 1), color='r',
            histtype='step', fill=False, linewidth=4, linestyle='--', label="200 MeV", align='left')
    print(f"200 MeV: {sum(np.diff(bins)*values)}")
    ax.set_xlabel("Number of tracks", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("Counts", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(kwargs["title"],
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    # ax.grid(linestyle = ':', linewidth = 2)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    # ax.grid()
    ax.set_xlim([0, data[1][:, 1].max()])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    # leg.get_frame().set_boxstyle('Square', pad=0.3)
    plt.savefig(f"./output/imgs/reconstruction/hist0{kwargs['name']}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_rec_effsmax(*args, **kwargs):
    data = args[0]
    if "name" in kwargs.keys():
        name = kwargs["name"]
    else:
        name = ""
    if "title" in kwargs.keys():
        title = kwargs["title"]
    else:
        title = ""
    x_range = np.linspace(0.2, 20, 100)*20
    # x_range = np.linspace(0.2, 20, 100)
    fig, ax = plt.subplots(figsize=(14, 10))
    # print(data.keys())
    c_indices = {70:0, 200:1} 
    l_indices = {70:0, 200:1} 
    for e_k, e_v in data.items():
        smax_eff_data = []
        nhit_keys = e_v.keys()
        for i in range(len(x_range)):
            arr = np.array([np.mean(np.array(e_v[n_k][i])) for n_k in nhit_keys])
            smax_eff_data.append(np.mean(arr))
        # print(np.shape(np.array(smax_eff_data)))
        # for d_k, d_v in e_v.items():
        #     smax_eff_data.append([[np.mean(v) for v in value[k]] for ])
        # print("pass")
        ax.plot(x_range, np.array(smax_eff_data) * 100, linewidth=3, 
                label=f"{e_k} MeV", c=['b', 'r'][c_indices[e_k]], 
                linestyle=['-', '--'][l_indices[e_k]])
        print(e_k)
    ax.axvline(190, linewidth=3, alpha=0.6, linestyle='dotted', c='k')
    ax.axvline(250, linewidth=3, alpha=0.6, linestyle='dotted', c='k')
    # ax.grid(linestyle = ':', linewidth = 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    ax.set_xlabel(r'S      (mrad)', loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("eff (%)", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    # ax.set_xlim([0, kwargs["maxx"] if "maxx" in kwargs.keys() else 300])
    ax.set_xlim([0, 300])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=24)
    ax.text(0.895, -0.12, f"max", transform=plt.gca().transAxes, size=FONT_SIZE*0.8, 
            ha='right', va='bottom', fontproperties=TIMES_BOLD)
    plt.savefig(f"./output/imgs/reconstruction/eff_smax{'_' if name else ''}{name}.png",
                bbox_inches='tight', dpi=300)
    plt.show()

def plot_rec_effsmax_all(*args, **kwargs):
    data = args[0]
    if "name" in kwargs.keys():
        name = kwargs["name"]
    else:
        name = ""
    if "title" in kwargs.keys():
        title = kwargs["title"]
    else:
        title = ""
    x_range = np.linspace(0.2, 20, 100)*20
    # x_range = np.linspace(0.2, 20, 100)
    fig, ax = plt.subplots(figsize=(14, 10))
    print(data.keys())
    for e_k, e_v in data.items():
        smax_eff_data = []
        nhit_keys = e_v.keys()
        for i in range(len(x_range)):
            arr = np.array([np.mean(np.array(e_v[n_k][i])) for n_k in nhit_keys])
            smax_eff_data.append(np.mean(arr))
        print(np.shape(np.array(smax_eff_data)))
        # for d_k, d_v in e_v.items():
        #     smax_eff_data.append([[np.mean(v) for v in value[k]] for ])
        # print("pass")
        ax.plot(x_range, np.array(smax_eff_data) * 100, linewidth=3, 
                label=f"{e_k} MeV")
    ax.grid(linestyle = ':', linewidth = 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    ax.set_xlabel("Smax (mrad)", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("eff (%)", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    # ax.set_xlim([0, kwargs["maxx"] if "maxx" in kwargs.keys() else 300])
    ax.set_xlim([0, 300])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=24)
    plt.savefig(f"./output/imgs/reconstruction/eff_smax{'_' if name else ''}{name}.png",
                bbox_inches='tight', dpi=300)
    # plt.show()

def plot_corl(*args, **kwargs):
    ALPIDE_coms = list(combinations([0, 1, 2, 3, 4, 5], 2))
    data = args[0]
    # ax200 = [plt.subplots(3, 5, figsize=(15, 9)) for i in range(2)]
    # print(data)
    for e_i, e in enumerate(np.unique(data["energy"].values)):
        # for mcs_i, mcs in enumerate(np.unique(data[data.energy == e]["mcs"].values)):
        for mcs_i, mcs in enumerate(np.unique(data[data.energy == e]["mcs"].values)):
            axs = [plt.subplots(3, 5, figsize=(18, 12)) for i in range(2)]
            for com_i, com in enumerate(np.unique(data[(data.energy == e) & (data.mcs == mcs)]["comb"].values)):
                pos_x = data[(data.mcs == mcs) & (data.energy == e) & (data.comb == com)]["x"].values
                pos_y = data[(data.mcs == mcs) & (data.energy == e) & (data.comb == com)]["y"].values
                # print(pos_x)
                if pos_x.size:
                    if com < 15:
                        # axs[0][1][int(com/5), int(com)%5].scatter(pos_x, pos_y, s=2, marker='s')
                        axs[0][1][int(com/5), int(com)%5].hist2d(pos_x, pos_y, bins=(20, 20), 
                                                                 range= [[12, 17], [12, 17]])
                        m_x, b_x, r_value_x, p_value_x, std_err_x =\
                        stats.linregress(pos_x, pos_y)
                        axs[0][1][int(com/5), int(com)%5].plot(np.linspace(12, 17, 20), 
                                            [b_x + m_x*px for px in np.linspace(12, 17, 20)],
                                            c='red')
                        print(f"energy : {e}, comb: {com}, slob: {m_x}")
                    else: 
                        # axs[1][1][int((com - 15)/5), int(com)%5]\
                        #     .scatter(pos_x, pos_y, s=2, marker='s')
                        axs[1][1][int((com - 15)/5), int(com)%5].hist2d(pos_x, pos_y, bins=(20, 20),
                                                                        range=[[5, 10], [5, 10]])
                        m_y, b_y, r_value_y, p_value_y, std_err_y =\
                        stats.linregress(pos_x, pos_y)
                        axs[1][1][int((com - 15)/5), int(com)%5].plot(np.linspace(5, 10, 20), 
                                            [b_y + m_y*px for px in np.linspace(5, 10, 20)],
                                            c='red')
                        print(f"energy : {e}, comb: {com}, slob: {m_y}")
            # axs[0][0].suptitle(f"Energy: {int(e)} x axis")
            # axs[1][0].suptitle(f"Energy: {int(e)} y axis")
            # for ax, com_ii in zip(axs[0][1].flat, range(len(ALPIDE_coms))):
            #     ax.set_xlim([12, 17])
            #     ax.set_ylim([12, 17])
            #     ax.set_title(f"{ALPIDE_coms[com_ii]}")
            # for ax, com_ii in zip(axs[1][1].flat, range(len(ALPIDE_coms))):
            #     ax.set_xlim([5, 10])
            #     ax.set_ylim([5, 10])
            #     ax.set_title(f"{ALPIDE_coms[com_ii]}")
            # axs[0][0].savefig(f"./output/imgs/reconstruction/correlation/e{int(e)}/mcs{mcs_i}_x.png", bbox_inches='tight', dpi=300)
            # axs[1][0].savefig(f"./output/imgs/reconstruction/correlation/e{int(e)}/mcs{mcs_i}_y.png", bbox_inches='tight', dpi=300)

def plot_corl_optz(*args, **kwargs):
    data = args[0]
    axis = args[1]
    x_fac = 1024./30
    y_fac = 512/13.8
    fig, ax = plt.subplots(figsize=(12, 12))
    if axis == "x":
        pos_x = data["x"].values*x_fac
        pos_y = data["y"].values*x_fac
        ax.hist2d(pos_x, pos_y, bins=(256, 256), range= [[pos_x.min(), pos_x.max()], 
                                                         [pos_y.min(), pos_y.max()]])
        m_x, b_x, r_value_x, p_value_x, std_err_x = stats.linregress(pos_x, pos_y)
        ax.plot(np.linspace(pos_x.min(), pos_x.max(), 20), 
                            [b_x + m_x*px for px in np.linspace(pos_x.min(), pos_x.max(), 20)],
                            c='red')
        ax.set_xlim([pos_x.min(), pos_x.max()])
        ax.set_ylim([pos_y.min(), pos_y.max()])
    else: 
        pos_x = data["x"].values*y_fac
        pos_y = data["y"].values*y_fac
        ax.hist2d(pos_x, pos_y, bins=(256, 256), range=[[5*y_fac, 10*y_fac], [5*y_fac, 10*y_fac]])
        m_y, b_y, r_value_y, p_value_y, std_err_y =\
        stats.linregress(pos_x, pos_y)
        ax.plot(np.linspace(5*y_fac, 10*y_fac, 20), 
                            [b_y + m_y*px for px in np.linspace(5*y_fac, 10*y_fac, 20)],
                            c='red')
        ax.set_xlim([5*y_fac, 10*y_fac])
        ax.set_ylim([5*y_fac, 10*y_fac])
    plt.show()
    # ax.savefig(f"./output/imgs/reconstruction/correlation/optz_{com_i}.png", bbox_inches='tight', dpi=300)

def plot_sim_beam(*args, **kwargs):
    data = args[0]
    hsv_mod = cm.get_cmap('jet', 300)
    num_col = 1024
    num_row = 512
    color_list = hsv_mod(np.linspace(0, 1.0, 30)).tolist()
    new_cmap = ListedColormap(color_list)
    fig, axs = plt.subplots(2, 3, figsize=(28, 10))
    for i in range(6):
        h = axs[int(i/3), i%3].hist2d(data[data.layerID == i]["posX"].values + 15,
                                  data[data.layerID == i]["posY"].values + 6.9,
                                  bins=(num_col, num_row), 
                                  cmap=new_cmap,
                                  range= [[0, 30], [0, 13.8]])
        # axs[int(i/3), i%3].set_xlim([0, 30]) 
        # axs[int(i/3), i%3].set_ylim([0, 15])
        axs[int(i/3), i%3].tick_params(axis='both', which='major', 
                                    width=2.5, length=10)
        axs[int(i/3), i%3].tick_params(axis='both', which='major', 
                                width=2.5, length=10)
        for x in axs[int(i/3), i%3].xaxis.get_major_ticks():
            x.label.set_fontproperties(TIMES_BOLD)
            x.label.set_fontsize(FONT_SIZE)
        for y in axs[int(i/3), i%3].yaxis.get_major_ticks():
            y.label.set_fontproperties(TIMES_BOLD)
            y.label.set_fontsize(FONT_SIZE)
        cbar = plt.colorbar(h[3], ax=axs[int(i/3), i%3], pad=0.008)
        for t in cbar.ax.get_yticklabels():
            t.set_fontproperties(TIMES_BOLD)
            t.set_fontsize(FONT_SIZE)
        # ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        axs[int(i/3), i%3].set_xlabel("posX (mm)", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        axs[int(i/3), i%3].set_ylabel("posY (mm)", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    plt.savefig(f"./imgs/Beam/simbeam_{kwargs['name']}.png", dpi=300, bbox_inches='tight')

def plot_rvalue_mcs(*args, **kwargs):
    data_dict = {"energy": [], "mcs": [], "entry": [], "slope": []}
    fig, ax = plt.subplots(2, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    ALPIDE_coms = list(combinations([0, 1, 2, 3, 4, 5], 2))
    data = args[0]
    energies = np.unique(data["energy"].values)
    for e_i, e in enumerate(energies):
        data_mcs = data[(data.energy == e) & (data.layer1 == 0) & (data.layer2 == 5)]
        data_coms = []
        for com_i, com in enumerate(ALPIDE_coms):    
            data_coms.append(data[(data.energy == e) & (data.layer1 == com[0]) & (data.layer2 == com[1])])
        for axis_i, axis in enumerate(["x", "y"]):
            ax[0].plot(data[data.energy == e]["mcs"]*1000,
                    [np.average([data_coms[i][data_coms[i].mcs == mcs][axis].values[0]**2 for i in range(len(data_coms))])\
                        for mcs in data[data.energy == e]["mcs"].values], label=f"{int(e)}MeV {axis.upper()}",
                    linewidth = 3)
        ax[1].plot(data_mcs["mcs"]*1000,
                data_mcs[f"numX"], label=f"{int(e)}MeV", linewidth=3)
        for mcs_i, mcs in enumerate(data_mcs["mcs"].values):
            data_dict["energy"].append(e)
            data_dict["mcs"].append(mcs)
            data_dict["entry"].append(data_mcs["numX"].values[mcs_i])
            if mcs_i == len(data_mcs["mcs"].index) - 1:
                data_dict["slope"].append((data_mcs["numX"].values[mcs_i] - data_mcs["numX"].values[mcs_i - 1])/\
                    (data_mcs["mcs"].values[mcs_i] - data_mcs["mcs"].values[mcs_i - 1]))
            else:
                data_dict["slope"].append((data_mcs["numX"].values[mcs_i] - data_mcs["numX"].values[mcs_i - 1])/\
                    (data_mcs["mcs"].values[mcs_i] - data_mcs["mcs"].values[mcs_i - 1]))
    
    ax[0].axvline(45, 0, 1, linestyle='--', linewidth=3, c='blue')
    ax[1].axvline(45, 0, 1, linestyle='--', linewidth=3, c='blue')
    ax[0].axvline(40, 0, 1, linestyle='--', linewidth=3, c='orange')
    ax[1].axvline(40, 0, 1, linestyle='--', linewidth=3, c='orange')
    ax[0].set_xlabel(f"$\Delta\Theta$ (m rad)", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax[0].set_ylabel(f"average R$^2$ value", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax[1].set_xlabel(f"$\Delta\\theta$ (m rad)", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax[1].set_ylabel(f"0-5 Enties", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    for x in ax[0].xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for x in ax[1].xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax[0].yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    for y in ax[1].yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    leg = ax[0].legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    leg = ax[1].legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"./output/imgs/reconstruction/correlation/r_value.png", dpi=300, bbox_inches='tight')
    return data_dict
    # plt.show()

def plot_sim_hit3D(*args, **kwargs):
    data = args[0]
    xx = np.linspace(-15, 15, 30)
    yy = np.linspace(-6.9, 6.9, 14)
    X, Y = np.meshgrid(xx, yy)
    labels = ["x mm", "y mm", "z mm"]
    facs = [1024/30, 512/13.8]
    lims = [[0, 30], [0, 15], [25, 181]]
    fig = plt.figure(figsize=(20, 15))
    ax = Axes3D(fig)
    ax.scatter(data["posX"], data["posZ"], data["posY"], s=100, c='r')
    for i in range(6):
        Z = np.ones(np.shape(xx))*(i*25 + 50)
        ax.plot_surface(X, Z, Y, alpha=0.2)
    ax.view_init(elev=8, azim=-10)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.zaxis.set_major_locator(ticker.NullLocator())
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
        x.label.set_visible(False)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    for z in ax.zaxis.get_major_ticks():
        z.label.set_fontproperties(TIMES_BOLD)
        z.label.set_fontsize(FONT_SIZE)
    ax.zaxis.set_tick_params(length=20, width=2)
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_linewidth(5)
    # ax.set_xticklabels(np.arange(6))  # Set text labels.
    ax.set_xlabel("\n\n row (1024 pixels)", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("\n\n\n ALPIDES", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_zlabel("\n\n column (512 pixels)", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.text(55, 28, -4, '                '.join([str(i) for i in range(6)]), color='k',
            size=FONT_SIZE, fontproperties=TIMES_BOLD) 
    ax.grid(False)
    plt.savefig(f"./output/imgs/simhit_{kwargs['name']}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_sim_rec_tracks(data): 
    data = data.sort_values(by=["MyTrackID", "layerID"])
    fig = plt.figure(figsize=(15, 12))
    ax = Axes3D(fig)    
    # xx = np.linspace(-15, 15, 30)
    # yy = np.linspace(-6.9, 6.9, 14)
    # X, Y = np.meshgrid(xx, yy)
    # for i in range(6):
    #     Z = np.ones(np.shape(xx))*(i*25 + 50)
    #     ax.plot_surface(X, Z, Y)
    for t_id in np.unique(data["MyTrackID"].values): 
        ax.plot3D(data[data.MyTrackID == t_id]["posX"].values, 
                  data[data.MyTrackID == t_id]["posZ"].values, 
                  data[data.MyTrackID == t_id]["posY"].values,
                linewidth=2, c='k')
    # for track in data:
    #     ax.scatter([j.data["posX"] for j in track], [j.data["posZ"] for j in track], [j.data["posY"] for j in track],
                # c='red', s=100)
    ax.view_init(elev=10, azim=-10)
    # view_init_list = [[0, 0], [0, -30]]
    ax.set_xlabel(f"X (mm)", fontproperties=TIMES_BOLD, fontsize=24, labelpad=30)
    ax.set_ylabel(f"Z (mm)", fontproperties=TIMES_BOLD, fontsize=24, labelpad=30)
    ax.set_zlabel(f"Y (mm)", fontproperties=TIMES_BOLD, fontsize=24, labelpad=30)

    ax.set_xlim([-15, 15])
    ax.set_ylim([40, 180])
    ax.set_zlim([-6.9, 6.9])
    # ax.set_xlim(lims[0])
    # ax.set_ylim(lims[2])
    # ax.set_zlim(lims[1])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(24)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(24)
    for z in ax.zaxis.get_major_ticks():
        z.label.set_fontproperties(TIMES_BOLD)
        z.label.set_fontsize(24)
    ax.grid(False)
    # plt.show()
    plt.savefig(f"{OUTPUT_DIR}/imgs/sim_rec_tracks_70MeV_400p.png", 
                dpi=300, bbox_inches='tight')

def plot_effsim_contour(*args, **kwargs):
    data = args[0]
    x = np.unique(data["mcs"].values)
    y = np.unique(data["smax"].values)
    y = y[y <= 0.06]
    X,Y = np.meshgrid(x, y)
    Z = np.zeros(np.shape(X))
    for row in range(np.shape(Z)[0]):
        for col in range(np.shape(Z)[1]):
            Z[row, col] = data[(data.mcs == X[row, col]) &\
                (data.smax == Y[row, col])]["eff"].values[0]
    X = X*1000
    Y = Y*1000
    fig, ax = plt.subplots(figsize=(12, 12))
    CS = ax.contourf(X, Y, Z, cmap='RdPu', vmin=0, vmax=1)
    clb = fig.colorbar(CS, ticks=[0.0 + i*0.2 for i in range(6)])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(24)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(24)
    ax.set_xlabel(r"Cone angle (mrad)", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_ylabel(r"S$_{max}$ (mrad)", fontproperties=TIMES_BOLD, fontsize=24)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.axvline(kwargs["sig"]*2e3, 0, 1, linestyle='--', linewidth=3)
    ax.annotate("2$\sigma_{\\theta_0}$", [kwargs["sig"]*2e3 + 0.1, 3], fontproperties=TIMES_BOLD, fontsize=24)
    for t in clb.ax.get_yticklabels():
            t.set_fontproperties(TIMES_BOLD)
            t.set_fontsize(FONT_SIZE)
    plt.savefig("./output/imgs/contourf_eff_200.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_pnoise(*args, **kwargv):
    data = args[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(len(data[0])), np.mean(data, axis=0), width=0.4, edgecolor='k',
           linewidth=2, color='b')
    # ax.bar(range(6), np.mean(data, axis=0))
    ax.set_xlabel(f"ALPIDE plane", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel(f"Average count", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_tick_params(which='major', width=1.5, length=18, direction="out")
    ax.yaxis.set_tick_params(which='minor', width=1, length=8, direction="out")
    ax.xaxis.set_tick_params(which='major', width=1.5, length=18, direction="out")
    # ax.grid(True, linestyle='--', linewidth=2)
    plt.savefig("./output/imgs/pnoise_hist.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_cnoise(*args, **kwargv):
    data = args[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    
    print(np.sort(np.unique(data["layerID"].values)))
    ax.bar(np.sort(np.unique(data["layerID"].values)) - 0.2*np.array([0, 0, 0, 0, 1, 0]), [
        len(data[(data.clusterSize == 1) & (data.layerID == i)].values)\
    for i in np.sort(np.unique(data["layerID"].values))], width=0.4, edgecolor='k',
           linewidth=2, color='aqua', label= '1-pixel')
    ax.bar(np.sort(np.unique(data["layerID"].values)) + 0.2*np.array([0, 0, 0, 0, 1, 0]), [
        len(data[(data.clusterSize == 2) & (data.layerID == i)].values)\
    for i in np.sort(np.unique(data["layerID"].values))], width=0.4,
           linewidth=2, color='r', edgecolor='k', label= '2-pixel', hatch='/')
    ax.set_xlabel(f"ALPIDE plane", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel(f"Cluster count", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_tick_params(which='major', width=1.5, length=18, direction="out")
    ax.yaxis.set_tick_params(which='minor', width=1, length=8, direction="out")
    ax.xaxis.set_tick_params(which='major', width=1.5, length=18, direction="out")
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)

    plt.savefig("./output/imgs/cnoise_hist.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_ncluster_layer(data, kind="cluster size", name="", title=""): 
    data_plot = []
    for l in np.unique(data["layerID"].values):
        data_plot.append(data[data.layerID == l]["clusterID"].values)
    fig, ax = plt.subplots(figsize=(14, 10))
    # ax.errorbar(range(6), 
    #         [np.mean(d) for d in data_plot],
    #         yerr=[np.std(d) for d in data_plot],
    #         marker='o', ms=16, ecolor='red', fmt=' ', elinewidth=4)
    plt.bar(range(6), [len(np.unique(d)) for d in data_plot])
    # ax.set_xticklabels([70, 100, 120, 150, 180, 200])
    ax.set_xlabel("ALPIDE layer", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel(f"Cluster count", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(title,
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    # ax.set_ylim([0, 15])
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_tick_params(which='major', width=1.5, length=18, direction="in")
    ax.yaxis.set_tick_params(which='minor', width=1, length=8, direction="in")
    ax.xaxis.set_tick_params(which='minor', width=1, length=8, direction="in")
    ax.xaxis.set_tick_params(which='major', width=1.5, length=18, direction="in")
    ax.grid(linestyle = ':', linewidth = 4)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    # ax.set_xticks([70, 100, 120, 150, 180, 200])
    # leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    # leg.get_frame().set_edgecolor('black')
    # leg.get_frame().set_boxstyle('Square', pad=0.3)
    plt.savefig(f"./output/imgs/mean_{kind.replace(' ', '_').lower()}_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_rec_effsmax_pmma(*args, **kwargs):
    data = args[0]
    if "name" in kwargs.keys():
        name = kwargs["name"]
    else:
        name = ""
    if "title" in kwargs.keys():
        title = kwargs["title"]
    else:
        title = ""
    # x_range = np.linspace(0.2, 20, 100)*20
    # x_range = np.linspace(0.2, 20, 100)
    fig, ax = plt.subplots(figsize=(14, 10))
    for d_i, d in enumerate(data):
        data_plot = []
        for smax_i, smax in enumerate(np.unique(d["smax"].values)):
            data_plot.append([smax*1000, d[(d.layerID == 5) & (d.smax == smax)].shape[0]])
        arr_data_plot = np.array(data_plot)
        ax.plot(arr_data_plot[:, 0], arr_data_plot[:, 1], label=["4.0 cm PMMA", "4.1 cm PMMA"][d_i], 
                linewidth=4)
    ax.set_xlim(0, 600)
    ax.grid(linestyle = ':', linewidth = 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    ax.set_xlabel("Smax (mrad)", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("Count", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    # ax.set_xlim([0, kwargs["maxx"] if "maxx" in kwargs.keys() else 300])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=24)
    plt.savefig(f"./output/imgs/reconstruction/eff_smax_pmma{'_' if name else ''}{name}.png",
                bbox_inches='tight', dpi=300)

def plot_rec_effmcs_pmma(*args, **kwargs):
    data = args[0]
    if "name" in kwargs.keys():
        name = kwargs["name"]
    else:
        name = ""
    if "title" in kwargs.keys():
        title = kwargs["title"]
    else:
        title = ""
    # x_range = np.linspace(0.2, 20, 100)*20
    # x_range = np.linspace(0.2, 20, 100)
    fig, ax = plt.subplots(figsize=(14, 10))
    for d_i, d in enumerate(data):
        data_plot = []
        for mcs_i, mcs in enumerate(np.unique(d["mcs"].values)):
            data_plot.append([mcs*1000, d[(d.layerID == 5) & (d.mcs == mcs)].shape[0]])
        arr_data_plot = np.array(data_plot)
        ax.plot(arr_data_plot[:, 0], arr_data_plot[:, 1], label=["4.0 cm PMMA", "4.1 cm PMMA"][d_i], 
                linewidth=4)
    ax.set_xlim(0, 1000)
    ax.grid(linestyle = ':', linewidth = 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    ax.set_xlim([0, 400])
    ax.set_xlabel(r"$\Delta$$\theta$ (mrad)", loc="right", 
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel("Count", loc="top",
                  fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    # ax.set_xlim([0, kwargs["maxx"] if "maxx" in kwargs.keys() else 300])
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    ax.set_title(title, fontproperties=TIMES_BOLD, fontsize=24)
    plt.savefig(f"./output/imgs/reconstruction/eff_mcs_pmma{'_' if name else ''}{name}.png",
                bbox_inches='tight', dpi=300)

def plot_HoverH3(data, hover=False, zoom=[], name="", title=""):
    hsv_mod = cm.get_cmap('jet', 300)
    num_col = 1024
    num_row = 512
    
    # valuesx, countsx = np.unique(data["posX"].values)
    # valuesy, countsy = np.unique(data["posX"].values)
    # count_max = countsx.max() if countsx.max() > countsy.max() else countsy.max()
    color_list = hsv_mod(np.linspace(0, 1.0, 30)).tolist()
    new_cmap = ListedColormap(color_list)
    for l_i, l in enumerate(np.unique(data["layerID"].values)):
        fig, ax = plt.subplots(figsize=(12, 10))
        data_layer = data[data.layerID == l]
        h = ax.hist2d(data_layer["posSubX"].values, data_layer["posSubY"].values, bins=(range(0, num_col, 1), 
                                                range(0, num_row, 1)), cmap=new_cmap)
        if hover:
            ax.scatter(data[:, 0], data[:, 1], alpha=0.5, c='red', s=10)
        if zoom:
            ax.set_xlim(zoom[0])
            ax.set_ylim(zoom[1])
        else: 
            ax.set_xlim([0, 1024])
            ax.set_ylim([0, 512])
            
        ax.tick_params(axis='both', which='major', 
                                    width=2.5, length=10)
        for x in ax.xaxis.get_major_ticks():
            x.label.set_fontproperties(TIMES_BOLD)
            x.label.set_fontsize(FONT_SIZE)
        for y in ax.yaxis.get_major_ticks():
            y.label.set_fontproperties(TIMES_BOLD)
            y.label.set_fontsize(FONT_SIZE)
        cbar = plt.colorbar(h[3], ax=ax, pad=0.008)
        for t in cbar.ax.get_yticklabels():
            t.set_fontproperties(TIMES_BOLD)
            t.set_fontsize(FONT_SIZE)
        ax.set_title(f"Layer {l_i}", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        ax.set_xlabel("X-position of pixel", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        ax.set_ylabel("Y-position of pixel", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)

        plt.savefig(f"./output/imgs/Beam/{name}_layer{l_i}_sub.png", dpi=300, bbox_inches='tight')

def plot_HoverH4(data, hover=False, energy=70, title=""):
    hsv_mod = cm.get_cmap('jet', 300)
    num_col = 1024
    num_row = 512
    
    color_list = hsv_mod(np.linspace(0, 1.0, 30)).tolist()
    new_cmap = ListedColormap(color_list)
    for l_i, l in enumerate(np.unique(data["layerID"].values)):
        fig, ax = plt.subplots(figsize=(12, 10))
        data_layer = data[data.layerID == l]
        h = ax.hist2d(data_layer["posSubX"].values, data_layer["posSubY"].values, bins=(range(0, num_col, 1), 
                                                range(0, num_row, 1)), cmap=new_cmap)
        if hover:
            ax.scatter(data[:, 0], data[:, 1], alpha=0.5, c='red', s=10)
        ax.set_xlim(cfg["experiment"][f"{energy} MeV"]["range"]["xsub"][l_i])
        ax.set_ylim(cfg["experiment"][f"{energy} MeV"]["range"]["ysub"][l_i])
            
        ax.tick_params(axis='both', which='major', 
                                    width=2.5, length=10)
        for x in ax.xaxis.get_major_ticks():
            x.label.set_fontproperties(TIMES_BOLD)
            x.label.set_fontsize(FONT_SIZE)
        for y in ax.yaxis.get_major_ticks():
            y.label.set_fontproperties(TIMES_BOLD)
            y.label.set_fontsize(FONT_SIZE)
        cbar = plt.colorbar(h[3], ax=ax, pad=0.008)
        for t in cbar.ax.get_yticklabels():
            t.set_fontproperties(TIMES_BOLD)
            t.set_fontsize(FONT_SIZE)
        ax.set_title(f"Layer {l_i}", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        ax.set_xlabel("X-position of pixel", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
        ax.set_ylabel("Y-position of pixel", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)

        plt.savefig(f"./output/imgs/Beam/{energy}MeV_layer{l_i}_sub.png", dpi=300, bbox_inches='tight')

def plot_hist_cluster2(data, kind="exp", name="", title="", sp=''): 
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['b', 'r']
    lines = ['-', '--']
    ens = ['70 MeV', '200 MeV']
    for d_i, d in enumerate(data):
        ax.hist(d, bins=range(min(d), max(d) + 1, 1), histtype='step', density=1,
                fill=False, color=colors[d_i], linewidth=2, linestyle=lines[d_i],
                label=ens[d_i])
    ax.set_xlabel("Cluster size", loc="right", 
                fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_xlim([0, 25])
    ax.set_ylabel("Count (norm.)", loc="top",
                fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title(title,
                fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    # ax.grid(linestyle = ':', linewidth = 2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', width=1.5, length=18, direction="in")
    ax.tick_params(which='minor', width=1, length=8, direction="in")
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    leg = ax.legend(prop={'fname': FNAME, 'size': FONT_SIZE})
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_boxstyle('Square', pad=0.3)
    plt.savefig(f"./output/imgs/cluster/sigma_cluster_{name}_2.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_num_hot():
    data = [3, 11, 8, 7, 12, 14]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(6), data, width=0.4, edgecolor='k',
           linewidth=2, color='b')
    # ax.bar(range(6), np.mean(data, axis=0))
    ax.set_xlabel(f"ALPIDE plane", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_ylabel(f"Number of pixels", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_tick_params(which='major', width=1.5, length=18, direction="out")
    ax.yaxis.set_tick_params(which='minor', width=1, length=8, direction="out")
    ax.xaxis.set_tick_params(which='major', width=1.5, length=18, direction="out")
    # ax.grid(True, linestyle='--', linewidth=2)
    plt.savefig("./output/imgs/hotpixels.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_6cpos(data: pd.DataFrame, axis="x"):
    fig, ax = plt.subplots(6, 1, figsize=(8, 14))
    for i in range(6):
        layer_data = data[data.layerID == i].sort_values(by=["eventID"])
        layer_data_plot = layer_data["posX" if axis=="x" else "posY"]
        ax[i].plot(np.arange(len(layer_data_plot.values)), layer_data_plot)
        ax[i].set_ylim([layer_data_plot.min(), layer_data_plot.max()])
    plt.show()

def plot_alpide_grid_track(track_data: pd.DataFrame, energy):
    layers = np.unique(track_data["layerID"].values)
    layers.sort()
    print(layers)
    for layer_i, layer in enumerate(layers):
        fig, ax = plt.subplots(figsize=(10, 10))
        x_sigma3 = int(2.1 * cfg["experiment"][f"{energy} MeV"]["sigmas"]["xsub"][int(layer)])
        y_sigma3 = int(2.1 * cfg["experiment"][f"{energy} MeV"]["sigmas"]["ysub"][int(layer)])
        ax.hist2d(
            track_data[track_data.layerID == int(layer)]["posX"].values,
            track_data[track_data.layerID == int(layer)]["posY"].values,
            bins=(range(512 - x_sigma3, 512 + x_sigma3, 1), range(256 - y_sigma3, 256 + y_sigma3, 1)),
            color='blue'
            )
        # ax.set_xlim([-x_sigma3, x_sigma3])
        # ax.set_ylim([-y_sigma3, y_sigma3])
        # ax.vlines(np.arange(-x_sigma3, x_sigma3), -y_sigma3, y_sigma3, 'k', linestyles='--', linewidth=0.5)
        # ax.hlines(np.arange(-y_sigma3, y_sigma3), -x_sigma3, x_sigma3, 'k', linestyles='--', linewidth=0.5)
        plt.show()

def plot_hit_events(file_path):
    files = os.listdir(file_path)
    files.sort()
    data = [[], []]
    for f in files:
        data_hit = cluster.exproot2array(path.join(file_path, f))
        data[0].append(int(f.split('_')[1].split('.')[0]))
        data[1].append(len((np.where(np.array(data_hit) > 0.0))[0]))
    #     event_id = int(f.split('_')[1].split('.')[0])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(*data)
    ax.set_xlabel("Event number")
    ax.set_ylabel("Entries")
    plt.show()

def plot_box_center(data: pd.DataFrame, kind="mu1", axis="x"):
    data = data[data.axis == axis]
    fig, ax = plt.subplots(figsize=(10, 8))
    layers = np.unique(data["layer"].values)
    layers.sort()
    ax.boxplot([data[data.layer == layer]["{}".format(kind)].values for layer in layers])
    ax.set_title("{} axis".format(axis.upper()))
    plt.show()

def plot_box_center(data: pd.DataFrame, kind="mu1", axis="x"):
    data = data[data.axis == axis]
    fig, ax = plt.subplots(figsize=(14, 8))
    layers = np.unique(data["layer"].values)
    layers.sort()
    bp = ax.boxplot([data[data.layer == layer]["{}".format(kind)].values for layer in layers])
    ax.set_title("{} axis".format(axis.upper()))
    ax.set_ylabel("$\mu$", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_xlabel("ALPIDE layer", fontproperties=TIMES_BOLD, fontsize=FONT_SIZE)
    ax.set_title("$\mu$ distribution in {} axis".format(axis), fontproperties=TIMES_BOLD,
                 fontsize=FONT_SIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_tick_params(which='major', width=1.5, length=18, direction="out")
    ax.yaxis.set_tick_params(which='minor', width=1, length=8, direction="out")
    ax.xaxis.set_tick_params(which='major', width=1.5, length=18, direction="out")
    for x in ax.xaxis.get_major_ticks():
        x.label.set_fontproperties(TIMES_BOLD)
        x.label.set_fontsize(FONT_SIZE)
    for y in ax.yaxis.get_major_ticks():
        y.label.set_fontproperties(TIMES_BOLD)
        y.label.set_fontsize(FONT_SIZE)
    plt.savefig("./newdata/output/imgs/centerboxes_{}.png".format(axis), 
                dpi=300, bbox_inches='tight')

def plot_center_zscore(data, axis="x"):
    new_data = data.copy()
    new_data = new_data[new_data.axis == axis]
    fig, axs = plt.subplots(3, 2, figsize=(18, 20))
    layers = np.unique(new_data["layer"].values)
    layers.sort()
    for layer_i, layer in enumerate(layers):
        z_scores = np.abs(zscore(new_data[new_data.layer==layer]["mu1"].values))
        if layer == 0:
            # print(new_data[new_data.layer==layer]["mu1"])
            print(z_scores)
        axs[int(layer/2), layer%2].bar(range(len(z_scores)), z_scores, label="{}".format(layer))
        axs[int(layer/2), layer%2].axhline(y=2, color='r', linestyle='--')
        leg = axs[int(layer/2), layer%2].legend(prop={'fname': FNAME, 'size': FONT_SIZE}, loc='upper right')
        for x in axs[int(layer/2), layer%2].xaxis.get_major_ticks():
            x.label.set_fontproperties(TIMES_BOLD)
            x.label.set_fontsize(FONT_SIZE)
        for y in axs[int(layer/2), layer%2].yaxis.get_major_ticks():
            y.label.set_fontproperties(TIMES_BOLD)
            y.label.set_fontsize(FONT_SIZE)
            axs[int(layer/2), layer%2].yaxis.set_minor_locator(AutoMinorLocator())
            axs[int(layer/2), layer%2].yaxis.set_tick_params(which='major', width=1.5, length=10, direction="out")
            axs[int(layer/2), layer%2].yaxis.set_tick_params(which='minor', width=1, length=5, direction="out")
            axs[int(layer/2), layer%2].xaxis.set_tick_params(which='major', width=1.5, length=10, direction="out")
    plt.suptitle("z-score distribution in {} axis".format(axis), fontproperties=TIMES_BOLD,
                 fontsize=FONT_SIZE)
    plt.savefig("./newdata/output/imgs/center_zscores_bars_{}.png".format(axis), 
                dpi=300, bbox_inches='tight')
    # print(z_scores)

def plot_6hist_center_line(data, axis="x", sub=False, cluster=False):
    data_label = "{}pos{}{}".format(
        "c" if cluster else "",
        "Sub" if sub else "",
        axis.upper() 
    )
    bin_width = 20 if cluster else 10
    new_data = data.copy()
    fig, axs = plt.subplots(3, 2, figsize=(18, 20))
    layers = np.unique(new_data["layerID"].values)
    layers.sort()
    for layer_i, layer in enumerate(layers):
        layer_cut = 1
        if layer == 4:
            layer_cut  = 2
        data_plot = new_data[(new_data.layerID==layer) & (new_data.clusterSize > layer_cut)]\
            ["{}".format(data_label)].values
        axs[int(layer/2), layer%2].hist(data_plot, bins=range(0, 1024, bin_width)\
                                                                  if axis == "x" else\
                                                                     range(
                                                                        0, 512, bin_width 
                                                                     ) 
                                        , label="{}".format(layer), histtype='step')
        leg = axs[int(layer/2), layer%2].legend(prop={'fname': FNAME, 'size': FONT_SIZE})
        if axis == "x":
            axs[int(layer/2), layer%2].axvline(x=512, color='r', linestyle='--')
        else:
            axs[int(layer/2), layer%2].axvline(x=256, color='r', linestyle='--')
        for x in axs[int(layer/2), layer%2].xaxis.get_major_ticks():
            x.label.set_fontproperties(TIMES_BOLD)
            x.label.set_fontsize(FONT_SIZE)
        for y in axs[int(layer/2), layer%2].yaxis.get_major_ticks():
            y.label.set_fontproperties(TIMES_BOLD)
            y.label.set_fontsize(FONT_SIZE)
            axs[int(layer/2), layer%2].yaxis.set_minor_locator(AutoMinorLocator())
            axs[int(layer/2), layer%2].yaxis.set_tick_params(which='major', width=1.5, length=10, direction="out")
            axs[int(layer/2), layer%2].yaxis.set_tick_params(which='minor', width=1, length=5, direction="out")
            axs[int(layer/2), layer%2].xaxis.set_tick_params(which='major', width=1.5, length=10, direction="out")
    plt.suptitle("{}beam{} distribution in {} axis"\
        .format("cluster " if cluster else "", " center" if sub else "", axis),
                 fontproperties=TIMES_BOLD,
                 fontsize=FONT_SIZE)
    plt.savefig("./newdata/output/imgs/hist6_{}{}{}.png"\
        .format("center_" if sub else "", "cluster_" if cluster else "", axis), 
                dpi=300, bbox_inches='tight')