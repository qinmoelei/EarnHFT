import pandas as pd
import numpy as np
import sys

sys.path.append(".")
from functools import partial
from model.net import Qnet
from multiprocessing import Pool
from env.low_level_env import Testing_env
import torch
import time


class actor:
    def __init__(self, model) -> None:
        self.trader = model
        self.device = "cpu"
        self.trader = self.trader.to(self.device)

    def act_test(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float(), 0).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]), 0).to(self.device)
        actions_value = self.trader.forward(x, previous_action,
                                            avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action


def collect_result(df, actor: actor, environment):
    specific_env = environment(df=df)
    done = False
    action_list = []
    reward_list = []
    s, info = specific_env.reset()
    while not done:
        action = actor.act_test(s, info)
        s_, r, done, info_ = specific_env.step(action)
        # tranjectory.append((s, info, action, r, s_, info_, done))
        s, info = s_, info_
        action_list.append(action)
        reward_list.append(r)
    action_list = np.array(action_list)
    reward_list = np.array(reward_list)
    require_money = specific_env.required_money
    final_balance = specific_env.final_balance
    return action_list, reward_list, require_money, final_balance


def collect_multiple_result(df_list, name_list, actor: actor, environment):
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool()
    func = partial(collect_result, actor=actor, environment=environment)
    result = pool.map(func, df_list)
    pool.close()
    pool.join()
    action_list_list = []
    reward_list_list = []
    require_money_list = []
    final_balance_list = []
    for i in range(len(result)):
        action_list, reward_list, require_money, final_balance = result[i]
        action_list_list.append(action_list)
        reward_list_list.append(reward_list)
        require_money_list.append(require_money)
        final_balance_list.append(final_balance)
    return name_list, action_list_list, reward_list_list, require_money_list, final_balance_list


if __name__ == "__main__":
    data = pd.read_feather("data/test/up.feather")
    data_list = [
        pd.read_feather("data/test/up.feather"),
        pd.read_feather("data/test/l_up.feather"),
        pd.read_feather("data/test/mid.feather"),
        pd.read_feather("data/test/l_down.feather"),
        pd.read_feather("data/test/down.feather"),
    ]
    features = np.load("data/selected_features.npy", allow_pickle=True)
    action_dim = 5
    net = Qnet(len(features), action_dim, 512)
    trader = actor(net)
    general_env = partial(
        Testing_env,
        tech_indicator_list=features,
        transcation_cost=0.001,
        back_time_length=1,
        max_holding_number=0.01,
        action_dim=action_dim,
    )
    # start_time = time.time()
    # collect_result(data, trader, general_env)
    # end_time = time.time()
    # print("it cost {} second to do one test".format(end_time - start_time))

    start_time = time.time()
    collect_multiple_result(data_list, ['up', "l_up", "mid", "l_down", "down"],
                            trader, general_env)
    end_time = time.time()
    print("it cost {} second to do 5 test".format(end_time - start_time))
