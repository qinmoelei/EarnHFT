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
    "--buffer_size",
    type=int,
    default=100000,
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--n_step",
    type=int,
    default=1,
    help="the number of step we have in the td error and replay buffer",
)
# RL & trading setting

parser.add_argument(
    "--action_dim",
    type=int,
    default=5,
    help="the number of action we have in the training and testing env",
)

parser.add_argument(
    "--back_time_length", type=int, default=20, help="the length of the holding period"
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the random seed for training and sample",
)

parser.add_argument(
    "--reward_scale",
    type=float,
    default=30,
    help="the scale factor we put in reward",
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
    "--batch_size",
    type=int,
    default=64,
    help="the number of transcation we learn at a time",
)
parser.add_argument("--update_times", type=int, default=1, help="the update times")
parser.add_argument("--gamma", type=float, default=1, help="the learning rate")
parser.add_argument(
    "--epsilon_init",
    type=float,
    default=1,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_min",
    type=float,
    default=0.01,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_decay_rate",
    type=float,
    default=0.00025,
    help="the coffient for decay",
)
parser.add_argument(
    "--update_frequency",
    type=int,
    default=128,
    help="the coffient for decay",
)

# general learning setting
parser.add_argument("--lr", type=float, default=0.00025, help="the learning rate")

parser.add_argument(
    "--num_sample",
    type=int,
    default=200,
    help="the overall number of sampling",
)
# trading setting
parser.add_argument(
    "--train_data_path",
    type=str,
    default="data/BTCTUSD/train",
    help="training data chunk",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCTUSD",
    help="training data chunk",
)

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
# log setting
parser.add_argument(
    "--result_path",
    type=str,
    default="result_risk",
    help="the path for storing the test result",
)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CDQN_rp(object):
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # log path
        self.model_path = os.path.join(
            args.result_path,
            args.dataset_name,
            "cdqn_rp",
            "seed_{}".format(self.seed),
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.grad_clip = 0.1

        # trading setting
        self.max_holding_number = args.max_holding_number
        self.action_dim = args.action_dim
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.reward_scale = args.reward_scale
        # RL setting
        self.update_counter = 0
        self.batch_size = args.batch_size
        self.update_times = args.update_times
        self.gamma = args.gamma
        self.epsilon_init = args.epsilon_init
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay_rate
        self.epsilon = self.epsilon_init
        self.target_freq_list = [5000, 8000, 10000, 20000]
        self.update_freq = args.update_frequency
        # replay buffer setting
        self.n_step = args.n_step
        self.buffer_size = args.buffer_size

        # general learning setting

        self.lr = args.lr
        self.num_sample = args.num_sample
        # data
        self.train_data_path = args.train_data_path
        self.chunk_num = 14400

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
        self.target_net = copy.deepcopy(self.eval_net)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.target_freq = self.target_freq_list[0]

    def update(
        self,
        states: torch.tensor,
        info: dict,
        actions: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        info_: dict,
        dones: torch.tensor,
    ):
        # TD error
        b = states.shape[0]
        q_eval = self.eval_net(
            states,
            info["previous_action"].float().unsqueeze(1),
            info["avaliable_action"],
        ).gather(1, actions)
        q_next = self.target_net(
            next_states,
            info_["previous_action"].float().unsqueeze(1),
            info_["avaliable_action"],
        ).detach()
        # since investigating is a open end problem, we do not use the done here to update
        q_target = rewards + torch.max(q_next, 1)[0].view(self.batch_size, 1) * (
            1 - dones
        )

        td_error = self.loss_func(q_eval, q_target)
        # KL divergence

        loss = td_error
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.update_counter > self.target_freq:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.update_counter = 0
            self.target_freq = self.target_freq_list[
                random.choices(
                    range(len(self.probability_list)),
                    weights=self.probability_list,
                    k=1,
                )[0]
            ]

        return (
            td_error.cpu(),
            torch.mean(q_eval.cpu()),
            torch.mean(q_target.cpu()),
            torch.mean(rewards.cpu()),
            torch.std(rewards.cpu()),
        )

    def act(self, state, info, epsilon):
        x = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float().to(self.device), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]).to(self.device), 0
        ).to(self.device)

        if np.random.uniform() > epsilon:
            actions_value = self.eval_net.forward(x, previous_action, avaliable_action)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = []
            for i in range(len(info["avaliable_action"])):
                if info["avaliable_action"][i] == 1:
                    action_choice.append(i)
            action = random.choice(action_choice)
        return action

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

    def train(self):
        step_counter_normal = 0
        c = 5000
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        # epoch_number = int(len(self.train_df) / self.chunk_length)
        epoch_number = 4
        random_position_list = random.choices(range(self.action_dim), k=self.num_sample)
        df_number = len(os.listdir(self.train_data_path)) - 1
        df_index_list = random.choices(range(df_number), k=self.num_sample)

        replay_buffer = Multi_step_ReplayBuffer_multi_info(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        reversed_list = self.target_freq_list[::-1]
        self.probability_list = np.exp(np.array(reversed_list) / c) / np.sum(
            np.exp(np.array(reversed_list) / c)
        )
        self.target_freq = self.target_freq_list[
            random.choices(
                range(len(self.probability_list)), weights=self.probability_list, k=1
            )[0]
        ]

        for sample in range(self.num_sample):
            if sample < math.ceil(self.num_sample / 3):
                reversed_list = self.target_freq_list[::-1]
                self.probability_list = np.exp(np.array(reversed_list) / c) / np.sum(
                    np.exp(np.array(reversed_list) / c)
                )
            if sample >= math.ceil(self.num_sample / 3) and sample < math.ceil(
                self.num_sample / 3 * 2
            ):
                self.probability_list = [1 / len(self.target_freq_list)] * len(
                    self.target_freq_list
                )
            if sample >= math.ceil(self.num_sample / 3 * 2):
                self.probability_list = np.exp(
                    np.array(self.target_freq_list) / c
                ) / np.sum(np.exp(np.array(self.target_freq_list) / c))

            df_index = df_index_list[sample]
            print("we are training with ", df_index)
            self.train_df = pd.read_feather(
                os.path.join(self.train_data_path, "df_{}.feather".format(df_index))
            )
            train_env = env_cqn(
                df=self.train_df,
                tech_indicator_list=self.tech_indicator_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                action_dim=self.action_dim,
                reward_scale=self.reward_scale,
                initial_action=random_position_list[sample],
            )
            s, info = train_env.reset()

            episode_reward_sum = 0
            while True:
                a = self.act(s, info, self.epsilon)

                s_, r, done, info_ = train_env.step(a)

                replay_buffer.add(s, info, a, r, s_, info_, done)
                episode_reward_sum += r
                step_counter_normal += 1
                self.update_counter += 1
                s, info = s_, info_

                if (
                    step_counter_normal > (self.batch_size + self.n_step)
                    and step_counter_normal % self.update_freq == 1
                ):
                    self.epsilon = (
                        self.epsilon * (1 - self.epsilon_decay)
                        if self.epsilon * (1 - self.epsilon_decay) > self.epsilon_min
                        else self.epsilon_min
                    )
                    for _ in range(self.update_times):
                        (
                            states,
                            infos,
                            actions,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        ) = replay_buffer.sample()
                        (
                            td_error,
                            q_eval,
                            q_target,
                            rewards_mean,
                            rewards_std,
                        ) = self.update(
                            states,
                            infos,
                            actions,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        )

                        self.writer.add_scalar(
                            tag="td_error",
                            scalar_value=td_error,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="q_eval",
                            scalar_value=q_eval,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="q_target",
                            scalar_value=q_target,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="rewards_mean",
                            scalar_value=rewards_mean,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="rewards_std",
                            scalar_value=rewards_std,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                if done:
                    break

            final_balance, required_money = (
                train_env.final_balance,
                train_env.required_money,
            )
            self.writer.add_scalar(
                tag="return_rate_train",
                scalar_value=final_balance / (required_money + 1e-12),
                global_step=sample,
                walltime=None,
            )
            self.writer.add_scalar(
                tag="final_balance_train",
                scalar_value=final_balance,
                global_step=sample,
                walltime=None,
            )
            self.writer.add_scalar(
                tag="required_money_train",
                scalar_value=required_money,
                global_step=sample,
                walltime=None,
            )
            self.writer.add_scalar(
                tag="reward_sum_train",
                scalar_value=episode_reward_sum,
                global_step=sample,
                walltime=None,
            )
            epoch_return_rate_train_list.append(
                final_balance / (required_money + 1e-12)
            )
            epoch_final_balance_train_list.append(final_balance)
            epoch_required_money_train_list.append(required_money)
            epoch_reward_sum_train_list.append(episode_reward_sum)
            if len(epoch_final_balance_train_list) == epoch_number:
                epoch_index = int((sample + 1) / epoch_number)
                mean_return_rate_train = np.mean(epoch_return_rate_train_list)
                mean_final_balance_train = np.mean(epoch_final_balance_train_list)
                mean_required_money_train = np.mean(epoch_required_money_train_list)
                mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
                self.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=mean_return_rate_train,
                    global_step=epoch_index,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="epoch_final_balance_train",
                    scalar_value=mean_final_balance_train,
                    global_step=epoch_index,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="epoch_required_money_train",
                    scalar_value=mean_required_money_train,
                    global_step=epoch_index,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="epoch_reward_sum_train",
                    scalar_value=mean_reward_sum_train,
                    global_step=epoch_index,
                    walltime=None,
                )
                epoch_path = os.path.join(
                    self.model_path, "epoch_{}".format(epoch_index)
                )
                if not os.path.exists(epoch_path):
                    os.makedirs(epoch_path)
                torch.save(
                    self.eval_net.state_dict(),
                    os.path.join(epoch_path, "trained_model.pkl"),
                )
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_required_money_train_list = []
                epoch_reward_sum_train_list = []


if __name__ == "__main__":
    args = parser.parse_args()
    agent = CDQN_rp(args)
    agent.train()
