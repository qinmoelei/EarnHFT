# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

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
from env.low_level_for_cqn import env_cqn
import copy
from RL.util.graph import get_test_contrast_curve
import re

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_path",
    type=str,
    default="result_risk/BTCTUSD/cdqn_rp/seed_12345/epoch_1",
    help="the path of test model",
)

parser.add_argument(
    "--action_dim",
    type=int,
    default=5,
    help="the number of action we have in the training and testing env",
)

parser.add_argument(
    "--back_time_length", type=int, default=20, help="the length of the holding period"
)


# network setting
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)
# RL training coffient

parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCTUSD",
    help="the name of the dataset, used for log",
)
parser.add_argument(
    "--valid_data_path",
    type=str,
    default="data/BTCTUSD/valid.feather",
    help="the path stored for train df list",
)
parser.add_argument(
    "--test_data_path",
    type=str,
    default="data/BTCTUSD/test.feather",
    help="the path stored for train df list",
)
parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0,
    help="the transcation cost for env",
)
parser.add_argument(
    "--max_holding_number",
    type=float,
    default=0.01,
    help="the max holding number of the coin",
)


class trader(object):
    def __init__(self, args) -> None:
        self.device = "cuda"
        # trading setting
        self.max_holding_number = args.max_holding_number
        self.action_dim = args.action_dim
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()
        # network
        self.test_path = args.test_path
        # pattern = r'hidden_nodes_(\d+)'
        # match = re.search(pattern, self.test_path)
        self.hidden_nodes = 128
        self.tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()

        self.n_state = len(self.tech_indicator_list)
        # network & loss function
        self.hidden_nodes = args.hidden_nodes
        self.eval_net = CQnet(
            self.n_state,
            self.action_dim,
            self.hidden_nodes,
            kernel_size=self.back_time_length,
        ).to(
            self.device
        )  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.eval_net.load_state_dict(
            torch.load(
                os.path.join(self.test_path, "trained_model.pkl"),
                map_location=torch.device("cpu"),
            )
        )
        self.eval_net = self.eval_net.to(self.device)
        # data
        self.valid_data_path = args.valid_data_path
        self.test_data_path = args.test_data_path

    def act_test(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float().to(self.device), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]).to(self.device), 0
        ).to(self.device)
        actions_value = self.eval_net.forward(x, previous_action, avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def load_model(self, epoch_path):
        self.eval_net.load_state_dict(
            torch.load(
                os.path.join(epoch_path, "trained_model.pkl"),
                map_location=torch.device(self.device),
            )
        )

    def test(self):
        self.load_model(self.test_path)
        for name, data_path in zip(
            ["valid", "test"], [self.valid_data_path, self.test_data_path]
        ):
            action_list = []
            reward_list = []
            self.test_df = pd.read_feather(data_path)
            self.test_env_instance = env_cqn(
                df=self.test_df,
                tech_indicator_list=self.tech_indicator_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                action_dim=self.action_dim,
                initial_action=0,
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
            if not os.path.exists(os.path.join(self.test_path, name)):
                os.makedirs(os.path.join(self.test_path, name))
            np.save(os.path.join(self.test_path, name, "action.npy"), action_list)
            np.save(os.path.join(self.test_path, name, "reward.npy"), reward_list)
            np.save(
                os.path.join(self.test_path, name, "final_balance.npy"),
                self.test_env_instance.final_balance,
            )
            np.save(
                os.path.join(self.test_path, name, "pure_balance.npy"),
                self.test_env_instance.pured_balance,
            )
            np.save(
                os.path.join(self.test_path, name, "require_money.npy"),
                self.test_env_instance.required_money,
            )
            np.save(
                os.path.join(self.test_path, name, "commission_fee_history.npy"),
                self.test_env_instance.comission_fee_history,
            )

            get_test_contrast_curve(
                self.test_df,
                os.path.join(self.test_path, name, "result.pdf"),
                reward_list,
                self.test_env_instance.required_money,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    agent = trader(args)
    agent.test()
