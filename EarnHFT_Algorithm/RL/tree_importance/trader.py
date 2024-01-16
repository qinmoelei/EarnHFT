import torch
import pandas as pd
import numpy as np
import sys

sys.path.append(".")
from model.net import Qnet,Qnet_L


class Trader:
    def __init__(self, N_state, N_action, hidden_nodes, model_path) -> None:
        self.model = Qnet(N_state, N_action, hidden_nodes)
        self.model.load_state_dict(torch.load(model_path))
        self.device = "cpu"
        self.model.to(self.device)

    def act_test(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float(), 0).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]), 0).to(self.device)
        actions_value = self.model.forward(x, previous_action,
                                           avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action


class Trader_L:
    def __init__(self, N_state, N_action, hidden_nodes, model_path) -> None:
        self.model = Qnet_L(N_state, N_action, hidden_nodes)
        self.model.load_state_dict(torch.load(model_path))
        self.device = "cpu"
        self.model.to(self.device)

    def act_test(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float(), 0).to(self.device)
        holding_length = torch.unsqueeze(
            torch.tensor([info["holding_length"]]).float().to(self.device),
            0).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]), 0).to(self.device)
        actions_value = self.model.forward(x, previous_action,
                                              holding_length, avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action