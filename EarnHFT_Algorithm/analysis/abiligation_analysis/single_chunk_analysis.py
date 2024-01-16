import pandas as pd
import numpy as np
import os
import re
import sys
from scipy import stats

sys.path.append(".")
from analysis.abiligation_analysis.util import calculate_holding_position_time
def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    lst.sort(key=alphanum_key)
#abliation 针对一个data chunk的指标 reward sum最终收敛值 average holding position time, 到converage所需要的step
#注意到这里面的不可使用time window来进行判断，因为每次差别过小
def calculate_result(setting_path,rolling_window_length=10):
    seed_path=os.path.join(setting_path,"seed_12345")
    update_conter_list=os.listdir(seed_path)
    sort_list(update_conter_list)
    update_conter_list.remove("log")
    reward_list=[]
    action_list_list=[]
    for epoch in update_conter_list:
        epoch_path=os.path.join(seed_path,epoch)
        reward=np.load(os.path.join(epoch_path,"test","reward.npy"))
        reward_list.append(np.sum(reward))
        action_list=np.load(os.path.join(epoch_path,"test","action.npy"))
        action_list_list.append(action_list)
    for start in range(rolling_window_length,len(reward_list)-rolling_window_length,1):
        reward_list_1=reward_list[start-rolling_window_length:start]
        reward_list_2=reward_list[start:start+rolling_window_length]
        _, p_value = stats.ks_2samp(reward_list_1, reward_list_2)
        if p_value>0.9:
            break
    converage_epoch_list=update_conter_list[start-rolling_window_length:start]
    converage_list=range(start-rolling_window_length,len(reward_list))
    print("converage_step",(start-rolling_window_length)*512)
    for start in converage_list:
        converage_action_list=action_list_list[start]
        average_holding_time=calculate_holding_position_time(converage_action_list)      
    print("average_holding_time",np.mean(average_holding_time))
    print("average reward list",np.mean(reward_list_1))
    numbers = re.findall(r'\d+', update_conter_list[start-rolling_window_length])
    









if __name__=="__main__":
    # df_path="ablation/GALAUSDT_df_9"
    
    
    df_path="ablation/ETHUSDT_df_27"
    setting_list=os.listdir(df_path)
    for setting_path in setting_list:
        print(setting_path)
        setting_path=os.path.join(df_path,setting_path)
        calculate_result(setting_path,rolling_window_length=40)