import pandas as pd
import numpy as np
import os
import re

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)
    
    
    
high_level_path="result_risk/GALAUSDT/high_level/seed_12345"



epoch_list=os.listdir(high_level_path)
epoch_list.remove("log")
valid_result=[]
sort_list(epoch_list)
for epoch in epoch_list[:80]:
    epoch_path = os.path.join(high_level_path, epoch)
    valid_path=os.path.join(epoch_path, "valid")
    return_rate=np.load(os.path.join(valid_path,"final_balance.npy"))/(np.load(os.path.join(valid_path,"require_money.npy")))
    valid_result.append(return_rate)


#rank 排名
rank_indices = np.argsort(valid_result)[::-1]

# 打印排名和对应的收益率
for rank, index in enumerate(rank_indices):
    epoch = epoch_list[index]
    epoch_path = os.path.join(high_level_path, epoch)
    test_path = os.path.join(epoch_path, "test")
    return_rate = np.load(os.path.join(test_path, "final_balance.npy")) / np.load(os.path.join(test_path, "require_money.npy"))
    print(f"Rank {rank+1}: Epoch {epoch}, Return Rate: {return_rate}")