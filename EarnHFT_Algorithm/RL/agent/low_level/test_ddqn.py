#Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
from torch.utils.tensorboard import SummaryWriter
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
import random
from tqdm import tqdm
import argparse
from model.net import *
import numpy as np
import torch
from torch import nn
import yaml
import pandas as pd
from env.low_level_env import Testing_env, Training_Env
import copy
from RL.util.graph import get_test_contrast_curve
from RL.util.episode_selector import start_selector, get_transformation_exp
import re
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
# path
parser.add_argument(
    "--test_path",
    type=str,
    default="result_risk/BTCUSDT/beta_-5.0_risk_bond_0.1/seed_12345/epoch_1",
    help="the path of test model",
)
parser.add_argument(
    "--test_df_path",
    type=str,
    default="data/BTCUSDT/valid",
    help="the path of test model",
)
# path
parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0.00015,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--max_holding_number",
    type=float,
    default=0.01,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--action_dim",
    type=int,
    default=5,
    help="the number of action we have in the training and testing env",
)
parser.add_argument("--back_time_length",
                    type=int,
                    default=1,
                    help="the length of the holding period")
parser.add_argument(
    "--reward_scale",
    type=float,
    default=30,
    help="the scale factor we put in reward",
)
parser.add_argument(
    "--initial_action",
    type=int,
    default=0,
    choices=[0,1,2,3,4],
    help="the initial action of the testing",
)


class trader(object):
    def __init__(self,args) -> None:
        self.device="cpu"
        #trading setting
        self.max_holding_number = args.max_holding_number
        self.action_dim = args.action_dim
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.reward_scale = args.reward_scale
        self.tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()
        self.initial_action=args.initial_action
        #network
        self.test_path=args.test_path
        # pattern = r'hidden_nodes_(\d+)'
        # match = re.search(pattern, self.test_path)
        self.hidden_nodes = 128
        self.input_dim=len(np.load("data/feature/second_feature.npy",allow_pickle=True))
  
        self.eval_net = Qnet(int(self.input_dim),
                             int(self.action_dim), int(self.hidden_nodes)).to(
                                 self.device)
        self.eval_net.load_state_dict(torch.load(os.path.join(self.test_path,"trained_model.pkl"),map_location=torch.device('cpu')))
        self.eval_net=self.eval_net.to(
                                 self.device)
        self.test_df_path=args.test_df_path
    def act_test(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float(), 0).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]), 0).to(self.device)
        actions_value = self.eval_net.forward(x, previous_action,
                                              avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action
    def test(self):
        label_list=os.listdir(self.test_df_path)
        for label in label_list:
            df_list=os.listdir(os.path.join(self.test_df_path,label))
            for df_path in df_list:
                self.test_df=pd.read_feather(os.path.join(self.test_df_path,label,df_path))
                self.test_env_instance = Testing_env(
                        df=self.test_df,
                        tech_indicator_list=self.tech_indicator_list,
                        transcation_cost=self.transcation_cost,
                        back_time_length=self.back_time_length,
                        max_holding_number=self.max_holding_number,
                        action_dim=self.action_dim,
                        initial_action=self.initial_action,
                    )
                s, info = self.test_env_instance.reset()
                done = False
                action_list = []
                reward_list = []
                while not done:
                    a = self.act_test(s, info)
                    s_, r, done, info_ = self.test_env_instance.step(a)
                    reward_list.append(r)
                    s = s_
                    info = info_
                    action_list.append(a)
                action_list = np.array(action_list)
                reward_list = np.array(reward_list)
                if not os.path.exists(os.path.join(self.test_path,"valid_multi", label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0])):
                    os.makedirs(os.path.join(self.test_path,"valid_multi", label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0]))
                np.save(os.path.join(self.test_path,"valid_multi", label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0], "action.npy"), action_list)
                np.save(os.path.join(self.test_path,"valid_multi", label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0], "reward.npy"), reward_list)
                np.save(
                    os.path.join(self.test_path,"valid_multi", label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0], "final_balance.npy"),
                    self.test_env_instance.final_balance,
                )
                np.save(
                    os.path.join(self.test_path,"valid_multi", label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0], "pure_balance.npy"),
                    self.test_env_instance.pured_balance,
                )
                np.save(
                    os.path.join(self.test_path,"valid_multi", label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0], "require_money.npy"),
                    self.test_env_instance.required_money,
                )
                # np.save(
                #     os.path.join(self.test_path, "valid_multi",label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0], "commission_fee_history.npy"),
                #     self.test_env_instance.comission_fee_history,
                # )
                # get_test_contrast_curve(
                #         self.test_df, os.path.join(self.test_path,"valid_multi", label,"initial_action_{}".format(self.initial_action),df_path.split(".")[0], "result.pdf"),
                #         reward_list, self.test_env_instance.required_money)

if __name__=="__main__":
    args = parser.parse_args()
    agent=trader(args)
    agent.test()