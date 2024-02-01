from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import os
import torch
import sys

# 由于在for循环中 np计算会存有一点点剩余 导致剩余的position不全是0 进而导致出现买不全的现象
sys.path.append(".")
from RL.util.graph import get_test_contrast_curve_high_level
from tool.demonstration import (
    making_multi_level_dp_demonstration,
    make_q_table,
    get_dp_action_from_qtable,
    make_q_table_reward,
)
import math
from env.low_level_env import Testing_env
from model.net import Qnet

high_level_tech_indicator_list = np.load("data/feature/minitue_feature.npy").tolist()
low_level_tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()

# tech_indicator_list=0
transcation_cost = 0.00
back_time_length = 1
max_holding_number = 0.01
action_dim = 5

# model_path_list=[
#     "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_1/trained_model.pkl",
#     "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_2/trained_model.pkl",
#     "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_3/trained_model.pkl",
#     "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_4/trained_model.pkl",
#     "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_5/trained_model.pkl",
# ]
model_path_list_dict = {
    0: [
        "result_risk/BTCUSDT/potential_model/initial_action_0/model_0.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_0/model_1.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_0/model_2.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_0/model_3.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_0/model_4.pth",
    ],
    1: [
        "result_risk/BTCUSDT/potential_model/initial_action_1/model_0.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_1/model_1.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_1/model_2.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_1/model_3.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_1/model_4.pth",
    ],
    2: [
        "result_risk/BTCUSDT/potential_model/initial_action_2/model_0.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_2/model_1.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_2/model_2.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_2/model_3.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_2/model_4.pth",
    ],
    3: [
        "result_risk/BTCUSDT/potential_model/initial_action_3/model_0.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_3/model_1.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_3/model_2.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_3/model_3.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_3/model_4.pth",
    ],
    4: [
        "result_risk/BTCUSDT/potential_model/initial_action_4/model_0.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_4/model_1.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_4/model_2.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_4/model_3.pth",
        "result_risk/BTCUSDT/potential_model/initial_action_4/model_4.pth",
    ],
}


class high_level_testing_env(Testing_env):
    def __init__(
        self,
        df: pd.DataFrame,
        high_level_tech_indicator_list=high_level_tech_indicator_list,
        low_level_tech_indicator_list=low_level_tech_indicator_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        action_dim=action_dim,
        early_stop=0,
        initial_action=0,
        model_path_list_dict=model_path_list_dict,
    ):
        self.device = "cpu"
        self.high_llevel_tech_inidcator_list = high_level_tech_indicator_list
        self.low_level_agent_list_dict = {}
        for key in model_path_list_dict:
            self.low_level_agent_list_dict[key] = []
            for model_path in model_path_list_dict[key]:
                model = Qnet(
                    int(len(low_level_tech_indicator_list)), int(action_dim), 128
                ).to("cpu")
                model.load_state_dict(
                    torch.load(
                        model_path,
                        map_location=torch.device("cpu"),
                    )
                )
                self.low_level_agent_list_dict[key].append(model)
        



        super(high_level_testing_env, self).__init__(
            df=df,
            tech_indicator_list=low_level_tech_indicator_list,
            transcation_cost=transcation_cost,
            back_time_length=back_time_length,
            max_holding_number=max_holding_number,
            action_dim=action_dim,
            early_stop=early_stop,
            initial_action=initial_action,
        )
        self.action_space = spaces.Discrete(len(self.low_level_agent_list_dict))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list),),
        )
        self.macro_action_history = []
        self.timestamp_history = []
        self.macro_state_history = []
        self.macro_reward_history=[]
        #log for model 
        self.chosen_model_history=[]

    def reset(self):
        self.macro_action_history = []
        self.timestamp_history = []
        self.state, self.info = super(high_level_testing_env, self).reset()
        high_level_state = self.data[self.high_llevel_tech_inidcator_list].values
        self.info["high_level_state"] = high_level_state
        self.chosen_model_history=[]
        
        return self.state, self.info

    def step(self, action):
        self.chosen_model = self.low_level_agent_list_dict[int(self.position/(self.max_holding_number/(self.action_dim-1)))][action]
        self.chosen_model_history.append(int(self.position/(self.max_holding_number/(self.action_dim-1)))*5+action)
        reward_mintue = 0
        while self.data.iloc[-1].timestamp.second != 59 and self.terminal == False:
            self.timestamp_history.append(self.data.iloc[-1].timestamp)
            self.macro_state_history.append(self.state)
            macro_action = self.pose_macro_action(self.state, self.info)
            self.macro_action_history.append(macro_action)
            self.state, reward, done, self.info = super(
                high_level_testing_env, self
            ).step(macro_action)
            self.macro_reward_history.append(reward)
            reward_mintue += reward
        if self.terminal == True:
            return self.state, reward_mintue, done, self.info
        else:
            self.timestamp_history.append(self.data.iloc[-1].timestamp)
            self.macro_state_history.append(self.state)
            macro_action = self.pose_macro_action(self.state, self.info)
            self.macro_action_history.append(macro_action)
            self.state, reward, done, self.info = super(
                high_level_testing_env, self
            ).step(macro_action)
            self.macro_reward_history.append(reward)
            reward_mintue += reward
            self.info["high_level_state"] = self.data[
                self.high_llevel_tech_inidcator_list
            ].values
        return self.state, reward_mintue, done, self.info

    def pose_macro_action(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float(), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]), 0
        ).to(self.device)
        actions_value = self.chosen_model.forward(x, previous_action, avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action


if __name__ == "__main__":
    df = pd.read_feather("data/BTCUSDT/valid.feather")
    reward_list = []
    test_env = high_level_testing_env(df)
    state, info = test_env.reset()
    done = False
    while not done:
        state, reward_mintue, done, info = test_env.step(1)
        reward_list.append(reward_mintue)
    (
        portfit_magine,
        final_balance,
        required_money,
        commission_fee,
    ) = test_env.get_final_return_rate(slient=False)
    get_test_contrast_curve_high_level(
        df, "test_high_level.pdf", reward_list, test_env.required_money
    )
