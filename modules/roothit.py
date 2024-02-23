import uproot
import pandas as pd
import numpy as np

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

def get_hit_simdata(file_path, num_events=10000, particles="both"):
    #exclude PDG 2212
    file_simulation = uproot.open(str(file_path))
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
    temp_pd_data["trackID"] = temp_pd_data["trackID"].astype('int64')
    if particles.lower() == "proton":
        temp_pd_data = temp_pd_data[temp_pd_data.trackID == 1]
    elif particles.lower() == "secondary":
        temp_pd_data = temp_pd_data[temp_pd_data.trackID!= 1]
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
    temp_pd_data2 = temp_pd_data2[temp_pd_data2.edep >= 0.0000036]
    # temp_pd_data2[temp_pd_data2.edep >= 0.1].to_csv("./logs/noise0.1.csv")
    return temp_pd_data2

def get_hit_expdata(file_path):
    data = []
    hit_keys = [f"Hitmaps/ALPIDE_{i}/h_hitmap_ALPIDE_{i};1" for i in range(6)]
    root_data = uproot.open(file_path)
    for key in hit_keys:
        if key in root_data.keys():
            data.append(root_data[key].to_numpy()[0].transpose()) 
    data_arr = np.array([i for i in data], dtype=object)
    temp_pd_data = pd.DataFrame(columns=["posX", "posY", "posZ", "layerID"])
    x_pos, y_pos, z_pos = [], [], []
    for i in range(len(data_arr)):
        if data_arr[i].ndim > 1:
            for j in range(len(data_arr[i])):
                x_pos.append(data_arr[i][j, 0])
                y_pos.append(data_arr[i][j, 1])
                z_pos.append(i)
        else:
            x_pos.append(data_arr[i][0])
            y_pos.append(data_arr[i][1])
            z_pos.append(i)
    temp_pd_data["posX"] = x_pos
    temp_pd_data["posY"] = y_pos
    temp_pd_data["posZ"] = z_pos
    temp_pd_data["posX"] = temp_pd_data["posX"]*30.0/1024 #mm
    temp_pd_data["posY"] = temp_pd_data["posY"]*13.8/512 #mm
    temp_pd_data["posZ"] = temp_pd_data["posZ"]*25.0 #mm
    temp_pd_data["layerID"] = z_pos 
    temp_pd_data["ID"] = temp_pd_data.index
    temp_pd_data["layerID"] = temp_pd_data["layerID"].astype("int")
    temp_pd_data["ID"] = temp_pd_data["ID"].astype("int")
    return temp_pd_data

def get_selsim_hit0(simdata, a=1.10, b=0.55):
    selsim_l0_data = simdata[((simdata.posX**2)/a**2 + (simdata.posY**2)/b**2 <= 1) & 
                             (simdata.layerID == 0)].copy()
    return selsim_l0_data

def get_hitexp_data(data):
    data_2 = data.copy()
    data_2.rename(columns={"cposX": "posX", "cposY": "posY", "clusterID": "ID"}, inplace=True)
    data_2["posX"] = data_2["posX"]*30.0/1024 #mm
    data_2["posY"] = data_2["posY"]*13.8/512 #mm
    posZ = data_2["layerID"].values*25.0 #mm
    data_2.insert(7, "posZ", posZ, allow_duplicates=False)
    return data_2
