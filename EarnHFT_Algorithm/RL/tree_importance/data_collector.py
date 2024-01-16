import pandas as pd
import numpy as np
import sys

sys.path.append(".")
from env.low_level_env import Testing_env
from RL.tree_importance.trader import Trader,Trader_L

df_list = [
    pd.read_feather("data/test/up.feather"),
    pd.read_feather("data/test/up_l.feather"),
    pd.read_feather("data/test/mid.feather"),
    pd.read_feather("data/test/down_l.feather"),
    pd.read_feather("data/test/down.feather"),
]
features = np.load("data/selected_features.npy", allow_pickle=True)
transcation_cost = 0.001
back_time_length = 1
max_holding_number = 0.01
action_dim = 5


def data_collector(df_list, hidden_nodes, model_path):
    X = []
    y = []
    for df in df_list:
        test_env = Testing_env(df=df,
                               tech_indicator_list=features,
                               transcation_cost=transcation_cost,
                               back_time_length=back_time_length,
                               max_holding_number=max_holding_number,
                               action_dim=action_dim)
        trader = Trader(len(features), action_dim, hidden_nodes, model_path)
        state, info = test_env.reset()
        done = False
        while not done:
            action = trader.act_test(state, info)
            X_state = state.tolist()
            X_state.append(info["previous_action"])
            X.append(X_state)
            y.append(action)
            state_, r, done, info_ = test_env.step(action)
            state, info = state_, info_
    return X, y

def data_collector_L(df_list, hidden_nodes, model_path):
    X = []
    y = []
    for df in df_list:
        test_env = Testing_env(df=df,
                               tech_indicator_list=features,
                               transcation_cost=transcation_cost,
                               back_time_length=back_time_length,
                               max_holding_number=max_holding_number,
                               action_dim=action_dim)
        trader = Trader_L(len(features), action_dim, hidden_nodes, model_path)
        state, info = test_env.reset()
        done = False
        while not done:
            action = trader.act_test(state, info)
            X_state = state.tolist()
            X_state.append(info["previous_action"])
            X_state.append(info["holding_length"])
            X.append(X_state)
            y.append(action)
            state_, r, done, info_ = test_env.step(action)
            state, info = state_, info_
    return X, y
if __name__ == "__main__":
    X, y = data_collector_L(
        df_list=df_list,
        hidden_nodes=512,
        model_path=
        "result/ddqn_L/ada_256_reward_scale_30_gamma_0.999_reject_coffient_0.01_advantage_coffient_0.5/seed_12345/epoch_4/trained_model.pkl"
    )
    np.save("RL/tree_importance/X_L.npy",X)
    np.save("RL/tree_importance/y_L.npy",y)
