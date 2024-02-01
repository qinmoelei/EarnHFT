import pandas as pd
import numpy as np
import sys
import os

sys.path.append(".")
from model.net import Qnet
from high_level_env_without_force_position import high_level_testing_env
from low_level_env import Testing_env
import torch
from RL.util.graph import get_test_contrast_curve,get_test_contrast_curve_high_level
high_level_tech_indicator_list = np.load("data/feature/minitue_feature.npy").tolist()
low_level_tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()

initial_action = 0
transcation_cost = 0.00
back_time_length = 1
max_holding_number = 0.01
action_dim = 5
model_path_list = [
    "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_1/trained_model.pkl",
    "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_2/trained_model.pkl",
    "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_3/trained_model.pkl",
    "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_4/trained_model.pkl",
    "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_5/trained_model.pkl",
]


class trader(object):
    def __init__(self) -> None:
        self.device = "cpu"
        # trading setting
        self.max_holding_number = max_holding_number
        self.action_dim = action_dim
        self.transcation_cost = transcation_cost
        self.back_time_length = back_time_length
        self.tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()
        self.initial_action = initial_action
        # network
        self.test_path = (
            "result_risk/boltzmann/beta_10.0_risk_bond_0.1/seed_12345/epoch_2"
        )
        # pattern = r'hidden_nodes_(\d+)'
        # match = re.search(pattern, self.test_path)
        self.hidden_nodes = 128
        self.input_dim = len(
            np.load("data/feature/second_feature.npy", allow_pickle=True)
        )

        self.eval_net = Qnet(
            int(self.input_dim), int(self.action_dim), int(self.hidden_nodes)
        ).to(self.device)
        self.eval_net.load_state_dict(
            torch.load(
                os.path.join(self.test_path, "trained_model.pkl"),
                map_location=torch.device("cpu"),
            )
        )
        self.eval_net = self.eval_net.to(self.device)

    def act_test(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float(), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]), 0
        ).to(self.device)
        actions_value = self.eval_net.forward(x, previous_action, avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action


if __name__ == "__main__":
    df = pd.read_feather("data/BTCTUSD/valid.feather").iloc[:100000]
    low_level_env = Testing_env(
        df=df,
        tech_indicator_list=low_level_tech_indicator_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        action_dim=action_dim,

        early_stop=0,
        initial_action=0,
    )
    low_level_agent=trader()
    high_level_env=high_level_testing_env(
        df=df,
        high_level_tech_indicator_list=high_level_tech_indicator_list,
        low_level_tech_indicator_list=low_level_tech_indicator_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        action_dim=action_dim,
        early_stop=0,
        initial_action=0,
        model_path_list=model_path_list,
    )
    s,info=low_level_env.reset()
    done = False
    action_list = []
    reward_list = []
    timestamp_list=[]
    state_list=[]
    while not done:
        timestamp_list.append(low_level_env.data.iloc[-1].timestamp)
        a = low_level_agent.act_test(s, info)
        state_list.append(s)
        s_, r, done, info_ = low_level_env.step(a)
        reward_list.append(r)
        s = s_
        info = info_
        action_list.append(a)
    action_list = np.array(action_list)
    reward_list = np.array(reward_list)
    state_list=np.array(state_list)
    np.save("low_level_action_list.npy",action_list)
    np.save("low_level_reward_list.npy",reward_list)
    np.save("low_level_timestamp_list.npy",timestamp_list)
    np.save("low_level_state_list.npy",state_list)
    get_test_contrast_curve(df=df, save_path="low_level.pdf", reward_list=reward_list, require_money=low_level_env.required_money)

    s,info=high_level_env.reset()
    done = False
    action_list = []
    reward_list = []
    while not done:
        s_, r, done, info_ = high_level_env.step(1)
        reward_list.append(r)
        action_list.append(a)
    action_list = np.array(high_level_env.macro_action_history)
    reward_list = np.array(reward_list)
    np.save("high_level_action_list.npy",action_list)
    np.save("high_level_reward_list.npy",reward_list)
    np.save("high_level_timestamp_list.npy",high_level_env.timestamp_history)
    np.save("high_level_state_list.npy",high_level_env.macro_state_history)
    get_test_contrast_curve_high_level(df=df, save_path="high_level.pdf", reward_list=reward_list, require_money=low_level_env.required_money)
    torch.save(
                    high_level_env.chosen_model.state_dict(),
                    os.path.join( "high_level_trained_model.pkl"),
                )