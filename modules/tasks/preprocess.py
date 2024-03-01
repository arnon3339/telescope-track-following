import pandas as pd
import numpy as np
import os
from os import path
import json

def merge_data(fs=[]):
    f_list = [f.split('/')[-1] for f in fs]
    en_list = [int(f.split('_')[1].split('MeV')[0]) for f in f_list]
    data_list = []
    for f_i, f in enumerate(fs):
        data_list.append(pd.read_csv(f, index_col=None))
        data_list[f_i].insert(0, "energy", np.ones(len(data_list[f_i].index))*en_list[f_i])
        data_list[f_i]["energy"] = data_list[f_i]["energy"].astype('int')
    dirname = path.dirname(fs[0])
    pd_data = pd.concat(data_list, ignore_index=True)
    pd_data = pd_data.iloc[:, ~pd_data.columns.str.contains('^Unnamed')]
    pd_data.to_csv(path.join(dirname, "data_merged.csv"), index=None)

def gen_sub_offset_data(f:str, offsets):
    def sub_data(value, layer, axis):
        return "{:.4f}".format(value - (offsets[axis][int(layer)] - (512 if axis == "x" else 256)))
    dirname = path.dirname(f)
    fname = f.split('/')[-1]
    data = pd.read_csv(f, index_col=None)
    data.insert(len(data.columns), "posSubX", data.\
        apply(lambda row: sub_data(row["posX"], row["layerID"], axis="x"), axis=1).astype("float64"))
    data.insert(len(data.columns), "posSubY", data.\
        apply(lambda row: sub_data(row["posY"], row["layerID"], axis="y"), axis=1).astype("float64"))
    data.insert(len(data.columns), "cposSubX", data.\
        apply(lambda row: sub_data(row["cposX"], row["layerID"], axis="x"), axis=1).astype("float64"))
    data.insert(len(data.columns), "cposSubY", data.\
        apply(lambda row: sub_data(row["cposY"], row["layerID"], axis="y"), axis=1).astype("float64"))
    data.to_csv(path.join(dirname, "sub{}".format(fname)), index=None)

def gen_phys_length(f:str):
    x_factor = 30.0/1024
    y_factor = 13.8/512
    dirname = path.dirname(f)
    fname = f.split('/')[-1]
    data = pd.read_csv(f, index_col=None)
    data.insert(len(data.columns), "posPhysSubX", data["posSubX"].\
        apply(lambda pix: "{:.4f}".format(pix*x_factor)).astype("float64"))
    data.insert(len(data.columns), "posPhysSubY", data["posSubY"].\
        apply(lambda pix: "{:.4f}".format(pix*y_factor)).astype("float64"))
    data.insert(len(data.columns), "cposPhysSubX", data["cposSubX"].\
        apply(lambda pix: "{:.4f}".format(pix*x_factor)).astype("float64"))
    data.insert(len(data.columns), "cposPhysSubY", data["cposSubY"].\
        apply(lambda pix: "{:.4f}".format(pix*y_factor)).astype("float64"))
    data.insert(len(data.columns), "posPhysX", data["posX"].\
        apply(lambda pix: "{:.4f}".format(pix*x_factor)).astype("float64"))
    data.insert(len(data.columns), "posPhysY", data["posY"].\
        apply(lambda pix: "{:.4f}".format(pix*y_factor)).astype("float64"))
    data.insert(len(data.columns), "cposPhysX", data["cposX"].\
        apply(lambda pix: "{:.4f}".format(pix*x_factor)).astype("float64"))
    data.insert(len(data.columns), "cposPhysY", data["cposY"].\
        apply(lambda pix: "{:.4f}".format(pix*y_factor)).astype("float64"))
    data.to_csv(path.join(dirname, "subphys_{}".format(fname)), index=None)

def astype_float2(f:str, columns=[]):
    dirname = path.dirname(f)
    fname = f.split('/')[-1]
    data = pd.read_csv(f, index_col=None)
    for column in columns:
        data[column] = data[column].apply(lambda x: "{:.2f}".format(x)).astype('float')
    data.to_csv(path.join(dirname, "float2{}".format(fname)), index=None)