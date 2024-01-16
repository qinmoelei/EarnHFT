from logging import raiseExceptions
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import sys


from torch.utils.data.sampler import (
    BatchSampler,
    SubsetRandomSampler,
    SequentialSampler,
)
from torch.distributions import Categorical
import copy


max_punish = 1e12


# Q network
# without holding length as input
class Qnet(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(N_ACTIONS + hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.fc3 = nn.Linear(1, N_ACTIONS)
        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(
        self,
        state: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        state_hidden = F.relu(self.fc1(state))
        previous_action_hidden = F.relu(self.fc3(previous_action))
        information_hidden = torch.cat([state_hidden, previous_action_hidden], dim=1)
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        masked_action = action + (avaliable_action - 1) * self.max_punish
        return masked_action


class Qnet_high_level(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Qnet_high_level, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)

    def forward(
        self,
        state: torch.tensor,
    ):
        state_hidden = F.relu(self.fc1(state))

        action = self.out(state_hidden)
        masked_action = action
        return masked_action


class Qnet_high_level_position(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Qnet_high_level_position, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc3 = nn.Linear(1, N_ACTIONS)
        self.fc2 = nn.Linear(N_ACTIONS + hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)

    def forward(
        self,
        state: torch.tensor,
        previous_action: torch.tensor,
    ):
        state_hidden = F.relu(self.fc1(state))
        previous_action_hidden = F.relu(self.fc3(previous_action))
        information_hidden = torch.cat([state_hidden, previous_action_hidden], dim=1)
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        masked_action = action
        return masked_action


# with holding length as input
class Qnet_L(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Qnet_L, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)

        self.fc2 = nn.Linear(2 * hidden_nodes, hidden_nodes)

        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.fc3 = nn.Linear(1, N_ACTIONS)

        self.fc4 = nn.Linear(1, hidden_nodes - N_ACTIONS)

        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(
        self,
        state: torch.tensor,
        previous_action: torch.tensor,
        holding_length: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        state_hidden = F.relu(self.fc1(state))
        previous_action_hidden = F.relu(self.fc3(previous_action))
        previous_length_hidden = F.relu(self.fc4(holding_length))
        information_hidden = torch.cat(
            [state_hidden, previous_action_hidden, previous_length_hidden], dim=1
        )
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        masked_action = action + (avaliable_action - 1) * self.max_punish
        return masked_action


# PPO network

# Actor


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(2 * hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(1, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.activate_func = nn.Tanh()  # Trick10: use tanh
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        orthogonal_init(self.out, gain=0.01)
        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(
        self,
        state: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        state_hidden = self.activate_func(self.fc1(state))
        previous_action_hidden = self.activate_func(self.fc3(previous_action))

        information_hidden = torch.cat([state_hidden, previous_action_hidden], dim=1)
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        logit = action + (avaliable_action - 1) * self.max_punish
        return logit


class Critic(nn.Module):
    def __init__(self, N_STATES, hidden_nodes):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(2 * hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(1, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, 1)
        self.activate_func = nn.Tanh()  # Trick10: use tanh
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        orthogonal_init(self.out, gain=0.01)

    def forward(
        self,
        state: torch.tensor,
        previous_action: torch.tensor,
    ):
        state_hidden = self.activate_func(self.fc1(state))
        previous_action_hidden = self.activate_func(self.fc3(previous_action))

        information_hidden = torch.cat([state_hidden, previous_action_hidden], dim=1)
        information_hidden = self.fc2(information_hidden)
        state_value = self.out(information_hidden)
        return state_value


# PPO lstm code reference:https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)

    return layer


class Actor_Critic_RNN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor_Critic_RNN, self).__init__()
        self.activate_func = nn.Tanh()  # Trick10: use tanh

        self.actor_rnn_hidden = None
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc_pa = nn.Linear(1, hidden_dim)

        self.actor_rnn = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.actor_fc2 = nn.Linear(hidden_dim, action_dim)

        self.critic_rnn_hidden = None
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc_pa = nn.Linear(1, hidden_dim)

        self.critic_rnn = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

        orthogonal_init(self.actor_fc1)
        orthogonal_init(self.actor_rnn)
        orthogonal_init(self.actor_fc2, gain=0.01)
        orthogonal_init(self.critic_fc1)
        orthogonal_init(self.critic_rnn)
        orthogonal_init(self.critic_fc2)
        self.register_buffer("max_punish", torch.tensor(max_punish))

    def actor(
        self,
        s: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        s = self.activate_func(self.actor_fc1(s))
        pa=self.activate_func(self.actor_fc_pa(previous_action))
   
        s=torch.cat([s, pa], dim=-1)
        # print(self.actor_rnn_hidden.shape)
        output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
        # for i in range(len(self.actor_rnn_hidden)):
            # print(self.actor_rnn_hidden[i].shape)
        logit = self.actor_fc2(output) + (avaliable_action - 1) * self.max_punish
        return logit

    def critic(self, s: torch.tensor,
        previous_action: torch.tensor,):
        s = self.activate_func(self.critic_fc1(s))
        pa=self.activate_func(self.critic_fc_pa(previous_action))
        s=torch.cat([s, pa], dim=-1)
        
        output, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
        value = self.critic_fc2(output)
        return value
#TODO do the convolutional neral network


class CQnet(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes,kernel_size=20):
        super(CQnet, self).__init__()
        
        # Convolution layer, input shape should be (B, N, T) for Conv1d
        self.conv1 = nn.Conv1d(N_STATES, hidden_nodes, kernel_size=20, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        # Note: You need to calculate the output size of the pooling layer
        # and use it as input size of fc1
        # For simplicity, I am assuming it as 'conv_output_size' here.
        conv_output_size = hidden_nodes // 2 # Replace this with actual output size after pooling
        self.fc1 = nn.Linear(conv_output_size + N_ACTIONS, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.fc3 = nn.Linear(1, N_ACTIONS)
        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(self, state:torch.tensor, previous_action:torch.tensor, available_action:torch.tensor):
        # Changing input shape to (B, N, T) for Conv1d
        state = state.permute(0, 2, 1)
        
        # Convolution and pooling
        state = F.relu(self.conv1(state))
        state = self.pool(state)
        
        # Flattening the output of the pooling layer
        state = state.view(state.size(0), -1)
        # Fully connected layers
        previous_action_hidden = F.relu(self.fc3(previous_action))
        information_hidden = torch.cat([state, previous_action_hidden], dim=1)
        information_hidden = F.relu(self.fc1(information_hidden))
        information_hidden = F.relu(self.fc2(information_hidden))
        action = self.out(information_hidden)
        
        # Masking
        masked_action = action + (available_action - 1) * self.max_punish
        
        return masked_action


if __name__ == "__main__":
    N_action = 11
    N_hidden = 32
    state = torch.randn(1,20, 66)
    previous_action = (
        torch.distributions.Binomial(10, torch.tensor([0.5] * 1))
        .sample()
        .float()
        .unsqueeze(0)
    )

    avaliable_action = torch.bernoulli(torch.Tensor(1, 11).uniform_(0, 1))
    # print("avaliable_action", avaliable_action)
    # holding_length = torch.randint(low=0, high=10, size=(1, 1)).float()
    # print(holding_length.shape)
    net = CQnet(66, 11, 32)
    # net_L = Qnet_L(66, 11, 32)
    # actor = Actor(66, 11, 32)
    # critic = Critic(66, 32)

    print(net(state, previous_action, avaliable_action).argmax(dim=1, keepdim=True))
    # print(actor(state, previous_action, avaliable_action))
    # print(critic(state, previous_action))
    # print(net_L(state, previous_action, holding_length, avaliable_action))



    # batch_size = 3
    # seq_len = 4
    # input_size = 8  # 2*hidden_dim, assuming hidden_dim = 4
    # hidden_dim = 4

    # # 创建 LSTM 层
    # lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)

    # # 创建输入张量
    # input = torch.randn(batch_size, seq_len, input_size)

    # # 通过 LSTM
    # output, (hn, cn) = lstm(input)
    # print(output.shape)
    # print(hn.shape)
    # print(cn.shape)

    # for i in range(5):
    #     output, (hn, cn) = lstm(input,(hn, cn))
    #     print("output",output.shape)
    #     print("hn",hn.shape)
    #     print("cn",cn.shape)