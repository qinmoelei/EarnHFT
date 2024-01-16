import pandas as pd
import numpy as np
import os
import re

def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)
high_level_path="result_risk/BTCTUSD/ppo/seed_12345"
epoch_list=os.listdir(high_level_path)
epoch_list.remove("log")
valid_result=[]
sort_list(epoch_list)
for epoch in epoch_list[:84]:
    epoch_path = os.path.join(high_level_path, epoch)
    valid_path=os.path.join(epoch_path, "test")
    return_rate=np.load(os.path.join(valid_path,"final_balance.npy"))/(np.load(os.path.join(valid_path,"require_money.npy"))+1e-12)
    valid_result.append(return_rate)
best_index=valid_result.index(max(valid_result))
print(valid_result)
best_epoch=epoch_list[best_index]
epoch_path = os.path.join(high_level_path, best_epoch)
test_path=os.path.join(epoch_path, "test")
return_rate=np.load(os.path.join(test_path,"final_balance.npy"))/(np.load(os.path.join(test_path,"require_money.npy"))+1e-12)
print(best_epoch)
print(return_rate)