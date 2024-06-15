import pandas as pd
import numpy as np
import os
import re


def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    lst.sort(key=alphanum_key)

analysis_path="result_risk/boltzmann"
par_list=os.listdir(analysis_path)
for par in par_list:
    single_par_return_rate_list=[]
    single_par_epoch_list=[]
    
    par_path=os.path.join(analysis_path,par)
    seed_list=os.listdir(par_path)
    for seed in seed_list:
        seed_path=os.path.join(par_path,seed)
        epoch_list=os.listdir(seed_path)
        epoch_list.remove("log")
        sort_list(epoch_list)
        for i in range(len(epoch_list)-1):
            epoch_path=os.path.join(seed_path,epoch_list[i],"valid")
            return_rate=np.load(os.path.join(epoch_path,"final_balance.npy"))/np.load(os.path.join(epoch_path,"require_money.npy"))
            single_par_return_rate_list.append(return_rate)
            epoch_list.append(i)
        print(par)
        print(epoch_list[single_par_return_rate_list.index(max(single_par_return_rate_list))])