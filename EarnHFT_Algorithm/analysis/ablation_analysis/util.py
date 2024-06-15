from scipy import stats
import numpy as np

# rewards1=[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
# rewards2=[2,4,6,8,10,2,4,6,8,10,2,4,6,8,10,2,4,6,8,10,2,4,6,8,10]


# ks_statistic, p_value = stats.ks_2samp(rewards1, rewards2)



def calculate_holding_position_time(action_list):
    holding_period_list=[0]
    previous_action=0
    holding_period=1
    for action in action_list:
        if action != previous_action:
            if previous_action!=0:
                holding_period_list.append(holding_period)
            previous_action=action
            holding_period=1
        else:
            holding_period+=1
    return np.mean(holding_period_list)