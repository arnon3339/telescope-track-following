import uproot
import pandas as pd
import numpy as np
import os

def cal_mean_epos(edep, pos):
    sum_pos = np.zeros(3)
    for i in range(3):
        for j in range(len(edep)):
            sum_pos[i] += edep[j]*pos[i, j]
    # if sum(edep) >= 0.1:
    #     with open("./logs/edep1.0.txt", "w+") as fs:
    #         fs.write("--------------------------")
    #         for i,j in zip(pos, ["x", "y", "z"]):
    #             fs.write(f"{j}: {i}")
    #         fs.write("--------------------------")
    return sum(edep), sum_pos/sum(edep)

def get_hit_data(file_path, num_events=10000):
    #exclude PDG 2212
    file_simulation = uproot.open(file_path)
    keys = file_simulation.keys()
    count_index = 0
    for i in range(len(keys)):
        if "Hits" == keys[i][:4]:
            count_index = i
            break
    temp_pd_data = pd.DataFrame(columns=["eventID", "posX", "posY", "posZ", "edep", "parentID", "layerID"])
    temp_pd_data["eventID"] = file_simulation[keys[count_index]]["eventID"].array().tolist()
    temp_pd_data["trackID"] = file_simulation[keys[count_index]]["trackID"].array().tolist()
    temp_pd_data["posX"] = file_simulation[keys[count_index]]["posX"].array().tolist()
    temp_pd_data["posY"] = file_simulation[keys[count_index]]["posY"].array().tolist()
    temp_pd_data["posZ"] = file_simulation[keys[count_index]]["posZ"].array().tolist()
    temp_pd_data["layerID"] = file_simulation[keys[count_index]]["level1ID"].array().tolist()
    temp_pd_data["edep"] = file_simulation[keys[count_index]]["edep"].array().tolist()
    temp_pd_data["parentID"] = file_simulation[keys[count_index]]["parentID"].array().tolist()
    temp_pd_data["PDGEncoding"] = file_simulation[keys[count_index]]["PDGEncoding"].array().tolist()
    temp_pd_data["PDGEncoding"] = temp_pd_data["PDGEncoding"].astype('int64')
    temp_pd_data = temp_pd_data[temp_pd_data.PDGEncoding != 2212]
    temp_pd_data = temp_pd_data[temp_pd_data.eventID < num_events]
    temp_pd_data["ID"] = temp_pd_data.index

    temp_pd_data2 = temp_pd_data.copy()
    repeat_pd_data = pd.DataFrame(columns=["eventID", "trackID", "posX", "posY", 
                                        "posZ", "edep", "parentID", "layerID", "ID", "PDGEncoding"])

    for index, row in temp_pd_data.iterrows():
        if len(temp_pd_data2[(temp_pd_data2.eventID == row["eventID"]) & (temp_pd_data2.trackID == row["trackID"]) & 
                        (temp_pd_data2.layerID == row["layerID"])].index) > 1:
            temp_temp_pd_data2 = temp_pd_data[(temp_pd_data.eventID == row["eventID"]) & 
                                            (temp_pd_data.trackID == row["trackID"]) & 
                                                (temp_pd_data.layerID == row["layerID"])]
            av_edep, av_pos = cal_mean_epos(np.array(temp_temp_pd_data2["edep"].tolist()), 
                                            np.array([temp_temp_pd_data2["posX"].tolist(),
                                            temp_temp_pd_data2["posY"].tolist(),
                                            temp_temp_pd_data2["posZ"].tolist()]))
            row_data = pd.DataFrame({"eventID": row["eventID"], "trackID": row["trackID"],"posX": av_pos[0], "posY": av_pos[1], 
                        "posZ": av_pos[2], "edep": av_edep, "parentID": row["parentID"], 
                        "layerID": row["layerID"], "ID": row["ID"], "PDGEncoding": row["PDGEncoding"]}, 
                        index=[row["ID"]])
            repeat_pd_data = pd.concat([repeat_pd_data, row_data], ignore_index=True)
            # repeat_pd_data = repeat_pd_data.append(row_data, ignore_index=True)
            for j in temp_temp_pd_data2["ID"].tolist():
                temp_pd_data2 = temp_pd_data2.drop(temp_pd_data2[temp_pd_data2["ID"] == j].index)

    temp_pd_data2 = pd.concat([temp_pd_data2, repeat_pd_data], ignore_index = True)
    temp_pd_data2 = temp_pd_data2.sort_values(by=["eventID", "trackID", "layerID"])
    temp_pd_data2.reset_index(drop=True, inplace=True)
    temp_pd_data2["ID"] = temp_pd_data2.index
    temp_pd_data2.layerID = temp_pd_data2.layerID.astype('int64')
    # data_proton_hits = data_proton_hits[data_proton_hits.edep >= 0.0000036]
    # temp_pd_data2[temp_pd_data2.edep >= 0.1].to_csv("./logs/noise0.1.csv")
    return temp_pd_data2

def get_hotpixel(name="col70MeV"):
    data_path = f"./data/experiment/{name}/"
    files = os.listdir(data_path)
    pixhot_arr_list = [[np.zeros((512, 1024)) for i in range(6)] for j in range(len(files))]
    for f_i, f in enumerate(files):
        root_data = uproot.open(f"{data_path}{f}")
        for k in range(6):
            hot_key = f"Hitmaps/ALPIDE_{k}/h_hotpixelmap_ALPIDE_{k};1"
            if hot_key in root_data.keys():
                hot_data = root_data[hot_key].to_numpy()[0]
                if hot_data.max() > 0:
                    print(f"Collecting file: {f}, layer: {k}")
                # pixhot_arr_list[f_i][k][:, :] = hot_data[:, :]