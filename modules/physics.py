import json
import numpy as np
import pandas as pd

with open("./config.json", 'r') as f:
    cfg = json.load(f)

def cal_pv(*args, **kwargs):
    return np.sqrt(2*kwargs["energy"])

def cal_HL_angle(*args, **kwargs):
    return 14.1*np.sqrt(kwargs["d"]/kwargs["len"])*\
        (1 + (1/9)*np.log10(kwargs["d"]/kwargs["len"]))/cal_pv(energy=kwargs["energy"])