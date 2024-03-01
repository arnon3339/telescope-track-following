import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from modules import mylplotlib
import pandas as pd
import json
from itertools import combinations

with open("./config.json", 'r') as f:
    cfg = json.load(f)

ALPIDE_coms = list(combinations([0, 1, 2, 3, 4, 5], 2))

def Gauss(x, x0, sigma, A):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

def Gauss_fit(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def Gauss_2fits(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1) + gauss(x,mu2,sigma2,A2)

def func_sigd(x, a, b, c):
    return a*np.log(b*x) + c

def get_est_hit_data(data, mean, lim):
    data.sort()
    values, counts = np.unique(data, return_counts=True)
    pd_data = pd.DataFrame({"values": values, "counts": counts})
    sel_pd_data = pd_data[(pd_data.values < lim[1]) & (pd_data.values > lim[0])]
    plt.plot(sel_pd_data["values"], sel_pd_data["counts"])
    parameters, covariance = curve_fit(Gauss, 
                                       sel_pd_data["values"].values, 
                                       sel_pd_data["counts"].values,
                                       p0=[sel_pd_data["counts"].max(), mean, 15]
                                       )
    print(parameters)
    # plt.show()
    

def get_hit_data(data):
    data.sort()
    values, counts = np.unique(data, return_counts=True)
    pd_data = pd.DataFrame({"values": values, "counts": counts})
    sel_pd_data = pd_data[(pd_data.values < 600) & (pd_data.values > 400)]
    plt.plot(sel_pd_data["values"], sel_pd_data["counts"])
    parameters, covariance = curve_fit(Gauss, 
                                       sel_pd_data["values"].values, 
                                       sel_pd_data["counts"].values,
                                       p0=[sel_pd_data["counts"].max(), 512, 15]
                                       )
    
    print(parameters)
    plt.show()

def get_calfit2(data):
    y_data = []
    unq_data = np.sort(np.unique(data))
    for d in unq_data:
        # print(np.where(data == d))
        y_data.append(len(np.where(data == d)[0]))
    y_data = np.array(y_data)

    mean = sum(unq_data*y_data)/sum(y_data)
    sigma = np.sqrt(sum(y_data*(unq_data - mean)**2)/sum(y_data))
    prms, cov = curve_fit(Gauss, unq_data, y_data, p0=[max(y_data), mean, sigma])
    mylplotlib.plot_beam1D([unq_data, y_data], prms)
    return sigma, mean

# def get_gfit(data, lim):
#     data_dict = {"layerID": [], "muX": [], "muY": [], "sigmaX": [], "sigmaY":[]}
#     data_2 = data[(data.posX > lim["x"][0]) & (data.posX < lim["x"][1]) & 
#                   (data.posY > lim["y"][0]) & (data.posY < lim["y"][1])].copy()
#     for i, layer in zip(range(len(np.unique(data["layerID"].values))), 
#                         np.unique(data["layerID"].values)):
#         data_dict["layerID"].append(layer)
#         mux = np.sum(data_2[data_2.layerID == layer]["posX"].values)\
#             /len(data_2[data_2.layerID == layer]["posX"].values)
#         sigmax = np.sqrt(np.sum((data_2[data_2.layerID == layer]["posX"].values - mux)**2)\
#             /len(data_2[data_2.layerID == layer]["posX"].values))
#         data_dict["muX"].append(mux)
#         data_dict["sigmaX"].append(sigmax)

#         muy = np.sum(data_2[data_2.layerID == layer]["posY"].values)\
#             /len(data_2[data_2.layerID == layer]["posY"].values)
#         sigmay = np.sqrt(np.sum((data_2[data_2.layerID == layer]["posY"].values - muy)**2)\
#             /len(data_2[data_2.layerID == layer]["posY"].values))
#         data_dict["muY"].append(muy)
#         data_dict["sigmaY"].append(sigmay)
#     return data_dict

def get_gfit(data, lim, expected, bi=False):
    data_dict = {"layerID": [], "muX": [], "muY": [], "sigmaX": [], "sigmaY":[], 
                 "paramsX": [], "paramsY": []}
    for i, layer in zip(range(len(np.unique(data["layerID"].values))), 
                        np.unique(data["layerID"].values)):
        data_2 = data[(data.posX > lim["x"][i][0]) & (data.posX < lim["x"][i][1]) & 
                    (data.posY > lim["y"][i][0]) & (data.posY < lim["y"][i][1]) & (data.layerID == layer)].copy()
        data_dict["layerID"].append(layer)
        valuesx, countsx = np.unique(data_2[data_2.layerID == layer]["posX"].values, return_counts=True)
        c_minx = countsx.min()
        countsx = countsx - c_minx
        if bi:
            params,cov = curve_fit(Gauss_2fits, valuesx, countsx, expected["x"][i])
        else:
            params,cov = curve_fit(gauss, valuesx, countsx, expected["x"][i])
        data_dict["muX"].append(np.dot(valuesx, countsx)/np.sum(countsx))
        data_dict["sigmaX"].append(np.sqrt(np.diag(cov)))
        data_dict["paramsX"].append(params)

        valuesy, countsy = np.unique(data_2[data_2.layerID == layer]["posY"].values, return_counts=True)
        c_miny = countsy.min()
        countsy = countsy - c_miny
        if bi:
            params,cov = curve_fit(Gauss_2fits, valuesy, countsy, expected["y"][i])
        else:
            params,cov = curve_fit(gauss, valuesy, countsy, expected["y"][i])
        data_dict["muY"].append(np.dot(valuesy, countsy)/np.sum(countsy))
        data_dict["sigmaY"].append(np.sqrt(np.diag(cov)))
        data_dict["paramsY"].append(params)
    return data_dict

def get_gfith(data, lim):
    data_dict = {"layerID": [], "muX": [], "muY": [], "sigmaX": [], "sigmaY":[]}
    for i, layer in zip(range(len(np.unique(data["layerID"].values))), 
                        np.unique(data["layerID"].values)):
        data_2 = data[(data.posX > lim["x"][i][0]) & (data.posX < lim["x"][i][1]) & 
                    (data.posY > lim["y"][i][0]) & (data.posY < lim["y"][i][1]) & (data.layerID == layer)].copy()
        data_dict["layerID"].append(layer)
        valuesx, countsx = np.unique(data_2[data_2.layerID == layer]["posX"].values, return_counts=True)
        c_minx = countsx.min()
        countsx = countsx - c_minx
        (mux, sigmax) = norm.fit(np.repeat(valuesx, countsx))
        data_dict["muX"].append(mux)
        data_dict["sigmaX"].append(sigmax)

        valuesy, countsy = np.unique(data_2[data_2.layerID == layer]["posY"].values, return_counts=True)
        c_miny = countsy.min()
        countsy = countsy - c_miny
        (muy, sigmay) = norm.fit(np.repeat(valuesy, countsy))
        data_dict["muY"].append(muy)
        data_dict["sigmaY"].append(sigmay)
    return data_dict

def get_gfit2g(data, expected):
    values, counts = np.unique(data, return_counts=True)
    params, covs = curve_fit(Gauss_2fits, values, counts, expected)
    return params

def get_gfitg(data, expected):
    values, counts = np.unique(data, return_counts=True)
    params, covs = curve_fit(Gauss, values, counts, expected)
    return params

def fit_sub(data, lims, expected):
    data_dict = {"layerID": [], "sigmaX": [], "sigmaY":[], 
                 "paramsX": [], "paramsY": []}
    data_fit = {"x": [], "y": []}
    for i, layer in zip(range(len(np.unique(data["layerID"].values))), 
                        np.unique(data["layerID"].values)):
        data_2 = data[(data.posX > lims[0]["x"][i][0]) & (data.posX < lims[0]["x"][i][1]) & 
                    (data.posY > lims[0]["y"][i][0]) & (data.posY < lims[0]["y"][i][1]) & (data.layerID == layer)].copy()
        valuesx, countsx = np.unique(data_2[data_2.layerID == layer]["posX"].values, return_counts=True)
        c_minx = countsx.min()
        countsx = countsx - c_minx
        params,cov = curve_fit(Gauss_2fits, valuesx, countsx, expected["x"][i])
        data_fit["x"].append([valuesx, countsx - gauss(valuesx, *params[3:])])

        valuesy, countsy = np.unique(data_2[data_2.layerID == layer]["posY"].values, return_counts=True)
        c_miny = countsy.min()
        countsy = countsy - c_miny
        params,cov = curve_fit(Gauss_2fits, valuesy, countsy, expected["y"][i])
        data_fit["y"].append([valuesy, countsy - gauss(valuesy, *params[3:])])
    
    for i in range(6):
        data_dict["layerID"].append(i)
        count_arrx = data_fit["x"][i][1].astype(np.int32)
        count_arrx[count_arrx < 0] = 0
        arrx = np.repeat(data_fit["x"][i][0], count_arrx)
        arrx = arrx[np.logical_and(arrx > lims[1]["x"][i][0], arrx < lims[1]["x"][i][1])]
        valuesx, countsx = np.unique(arrx, return_counts=True)
        params, cov = curve_fit(gauss, valuesx, countsx, expected["x"][i][:3])
        data_dict["sigmaX"].append(np.sqrt(np.diag(cov)))
        data_dict["paramsX"].append(params)

        count_arry = data_fit["y"][i][1].astype(np.int32)
        count_arry[count_arry < 0] = 0
        arry = np.repeat(data_fit["y"][i][0], count_arry)
        arry = arry[np.logical_and(arry > lims[1]["y"][i][0], arry < lims[1]["y"][i][1])]
        valuesy, countsy = np.unique(arry, return_counts=True)
        params, cov = curve_fit(gauss, valuesy, countsy, expected["y"][i][:3])
        data_dict["sigmaY"].append(np.sqrt(np.diag(cov)))
        data_dict["paramsY"].append(params)
        
    return data_dict

def get_mulevt_tract_eff(*args, **kwargs):
    data = args[0]
    data_dict = {e: {} for e in data.keys()}
    smaxs = np.linspace(0.2, 20, 100)
    for e_i, (e_k, e_v) in enumerate(data.items()):
        for n_i, (n_k, n_v) in enumerate(e_v.items()):
            data_dict[e_k][n_k] = []
            for smax_i in range(smaxs.size):
                data_ni = [n_v[nn_i][smax_i] for nn_i in range(len(n_v))]
                dd_eff = []
                for dd_i, dd in enumerate(data_ni):
                    # print(dd)
                    if not hasattr(dd, "__len__"):
                        dd_eff.append(1 if dd == 6 else 0)
                    elif not len(dd):
                        dd_eff.append(0)
                    else:
                        if any(dd):
                            dd_data = [ddd for ddd in dd if ddd != 0]
                            dd_eff.append(sum([1 for ddd in dd_data if ddd == 6])/len(dd_data))
                        else:
                            dd_eff.append(0)
                dd_eff_arr = np.zeros(len(dd_eff) - cfg["experiment"][f"{e_k} MeV"]["n_fevts"][f"{n_k}"])
                non_zero_arr = [eff_dd for eff_dd in dd_eff if eff_dd != 0]
                dd_eff_arr[:len(non_zero_arr)] = non_zero_arr
                data_dict[e_k][n_k].append(dd_eff_arr)
    return data_dict
# def get_gfit(data, lim):
#     data_2 = data[np.logical_and(data > lim[0], data < lim[1])].copy()
#     data_2_unique = np.unique(data_2, return_counts=True)
#     data_x = data_2_unique[0]
#     data_y = data_2_unique[1] - data_2_unique[1].min()
#     parameters, covariance = curve_fit(Gauss, data_x, data_y)
#     # (mu, sigma) = norm.fit(data_2)
#     fit_y = Gauss_fit(data_x, parameters[0], parameters[1])
#     return np.array([data_x, fit_y]).transpose()
def find_correlation(data):
    data_dict = {
        "energy": [],
        "mcs": [],
        "eventID": [],
        "trackID": [],
        "comb": [],
        "x": [],
        "y": [],

    }
    for e_i, e in enumerate(np.unique(data["energy"].values)):
        for mcs_i, mcs in enumerate(np.unique(data["MSCangle"].values)):
            for com_i, com in enumerate(ALPIDE_coms*2):
                for evt_i, evt in enumerate(np.unique(data["eventID"].values)):
                    data_x = data[(data.energy == e) & (data.layerID == com[0]) \
                        & (data.eventID == evt) & (data.MSCangle == mcs)]
                    track_ids = np.unique(data_x["MyTrackID"])
                    data_y = data[(data.energy == e) & (data.layerID == com[1]) \
                        & (data.eventID == evt) & (data.MSCangle == mcs) & (data.MyTrackID.isin(track_ids))]
                    # data_y = data[(data.layerID == com[1]) & (data.MyTrackID.isin(track_ids)) & (data.eventID == evt)]
                    for tid_i, tid in enumerate(track_ids): 
                        if len(data_y[data_y.MyTrackID == tid].index):
                            data_dict["energy"].append(e)
                            data_dict["mcs"].append(mcs)
                            data_dict["eventID"].append(evt)
                            data_dict["comb"].append(com_i)
                            data_dict["trackID"].append(tid)
                            if com_i < int(len(ALPIDE_coms*2)/2):
                                data_dict["x"].append(data_x[data_x.MyTrackID == tid]["posX"].values[0])
                                data_dict["y"].append(data_y[data_y.MyTrackID == tid]["posX"].values[0])
                            else:
                                data_dict["x"].append(data_x[data_x.MyTrackID == tid]["posY"].values[0])
                                data_dict["y"].append(data_y[data_y.MyTrackID == tid]["posY"].values[0])
                    print(f"Finished energy: {e}, mcs: {mcs}, eventID: {evt}")
    return data_dict

def clear_outliers(data):
    q3 = np.percentile(data, 75)
    q1 = np.percentile(data, 25)

    iqr = q3 - q1
    threshold = 1.5
    return data[np.where((data < q3 + iqr*threshold) & (data > q1 - iqr*threshold))[0]]