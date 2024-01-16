import pandas as pd
import numpy as np
import os
import re
import torch
import argparse

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--root_path",
    type=str,
    default="result_risk/BTCUSDT",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="result_risk/BTCUSDT/potential_model",
    help="the number of transcation we store in one memory",
)


def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)

#TODO change the picking process as long with the position included


args = parser.parse_args()

def get_best_model(label_name,initial_position,root_path=args.root_path):
    coffient_list = os.listdir(root_path)
    if "potential_model" in coffient_list:
        coffient_list.remove("potential_model")
    result_dict = {}
    for coffient in coffient_list:
        coffient_path = os.path.join(root_path, coffient, "seed_12345")
        epoch_list = os.listdir(coffient_path)
        epoch_list.remove("log")
        sort_list(epoch_list)
        for epoch in epoch_list:
            epoch_path = os.path.join(coffient_path, epoch)
            target_path = os.path.join(epoch_path, "valid_multi", label_name,initial_position)
            return_rate_list = []
            normalized_return_rate_list = []
            reward_sum_list=[]
            df_list = os.listdir(target_path)
            sort_list(df_list)
            for df in df_list:
                result_path = os.path.join(target_path, df)
                return_rate = np.load(
                    os.path.join(result_path, "final_balance.npy"), allow_pickle=True
                ) / (np.load(
                    os.path.join(result_path, "require_money.npy"), allow_pickle=True
                )+1e-12)
                return_rate_list.append(return_rate)
                normalized_return_rate_list.append(
                    return_rate
                    / len(
                        np.load(os.path.join(result_path, "action.npy"), allow_pickle=True)
                    )
                )
                rewards=np.load(
                    os.path.join(result_path, "reward.npy"), allow_pickle=True
                )
                reward_sum_list.append(np.sum(rewards))
            result_dict["{}_{}".format(coffient, epoch)] = {
                "return_rate_list": return_rate_list,
                "normalized_return_rate_list": normalized_return_rate_list,
                "average_return_rate": np.mean(return_rate_list),
                "average_normalized_return_rate": np.mean(normalized_return_rate_list),
                "reward_sum_list":reward_sum_list,
            }


    def find_max_average_return_rate(dictionary):
        max_average_return_rate = float("-inf")
        best_index = list(dictionary.keys())[0]
        for index in dictionary.keys():
            sub_dictionary = dictionary[index]
            average_return_rate = sub_dictionary["average_return_rate"]

            if average_return_rate > max_average_return_rate:
                max_average_return_rate = average_return_rate
                best_index = index

        return best_index
    
    
    def find_max_reward_sum(dictionary):
        max_average_return_rate = float("-inf")
        best_index = list(dictionary.keys())[0]
        for index in dictionary.keys():
            sub_dictionary = dictionary[index]
            average_return_rate = np.mean(sub_dictionary["reward_sum_list"])

            if average_return_rate > max_average_return_rate:
                max_average_return_rate = average_return_rate
                best_index = index

        return best_index


    print(find_max_reward_sum(result_dict))
    return find_max_reward_sum(result_dict)

if __name__=="__main__":
    args = parser.parse_args()
    root_path=args.root_path
    save_path=args.save_path
    initial_position_list=["initial_action_0","initial_action_1","initial_action_2","initial_action_3","initial_action_4"]
    for initial_position in initial_position_list:
        model_path_list=[]
        for i in range(5):
            print("label_{}".format(i))
            index_model=get_best_model(label_name="label_{}".format(i),initial_position=initial_position)
            elements = index_model.split("_epoch_")
            para=elements[0]
            epoch_number=elements[1]

            model_path_list.append(os.path.join(root_path,para,"seed_12345","epoch_{}".format(epoch_number),"trained_model.pkl"))
        model_path_list=list((model_path_list))
        model_list=[torch.load(model_path
        ) for model_path in model_path_list]
        if not os.path.exists(os.path.join(save_path,initial_position)):
            os.makedirs(os.path.join(save_path,initial_position),exist_ok=True)
        for i in range(len(model_list)):
            torch.save(model_list[i],os.path.join(save_path,initial_position,"model_{}.pth".format(i)))