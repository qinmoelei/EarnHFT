import pandas as pd
import numpy as np
import os
import re
def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)
def calculate_rank(path):
    #输入为seed_path,其子集应该为epoch path 最终返回的应该是两个rank的正常的correlation
    valid_value_list=[]
    test_value_list=[]
    epoch_list=os.listdir(path)
    epoch_list.remove("log")
    sort_list(epoch_list)
    for epoch in epoch_list:
        epoch_path = os.path.join(path, epoch)
        valid_path=os.path.join(epoch_path, "valid")
        return_rate=np.load(os.path.join(valid_path,"final_balance.npy"))/(np.load(os.path.join(valid_path,"require_money.npy")))
        valid_value_list.append(return_rate)

        test_path=os.path.join(epoch_path, "test")
        return_rate=np.load(os.path.join(test_path,"final_balance.npy"))/(np.load(os.path.join(test_path,"require_money.npy")))
        test_value_list.append(return_rate)
    
