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
from env.low_level_env import Testing_env, Training_Env
from env.high_level_env import high_level_testing_env
import copy
from RL.util.graph import get_test_contrast_curve_high_level
from RL.util.episode_selector import (
    start_selector,
    get_transformation_even_risk,
    get_transformation_even_based_boltzmann_risk,
    get_transformation_even_based_sigmoid_risk,
)
import re

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
# replay buffer coffient
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
    "--back_time_length", type=int, default=1, help="the length of the holding period"
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
    default=10,
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
    "--tau", type=float, default=0.005, help="soft update the target network"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="the number of transcation we learn at a time",
)
parser.add_argument("--update_times", type=int, default=5, help="the update times")
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
    default=0.1,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_step",
    type=float,
    default=5e4,
    help="the coffient for decay",
)
parser.add_argument(
    "--target_freq",
    type=int,
    default=512,
    help="the number of sampling during one epoch",
)
# general learning setting
parser.add_argument("--lr_init", type=float, default=1e-2, help="the learning rate")
parser.add_argument("--lr_min", type=float, default=5e-4, help="the learning rate")
parser.add_argument("--lr_step", type=float, default=5e4, help="the learning rate")
parser.add_argument(
    "--num_sample",
    type=int,
    default=100,
    help="the overall number of sampling",
)
# trading setting
parser.add_argument(
    "--train_data_path",
    type=str,
    default="data/BTCUSDT/train.feather",
    help="training data chunk",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
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


def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
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
            "high_level_without_position",
            "seed_{}".format(self.seed),
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)

        # trading setting
        assert args.dataset_name in ["BTCUSDT", "ETHUSDT", "GALAUSDT"]
        if args.dataset_name == "BTCUSDT":
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
        elif args.dataset_name == "ETHUSDT":
            model_path_list_dict = {
                0: [
                    "result_risk/ETHUSDT/potential_model/initial_action_0/model_0.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_0/model_1.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_0/model_2.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_0/model_3.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_0/model_4.pth",
                ],
                1: [
                    "result_risk/ETHUSDT/potential_model/initial_action_1/model_0.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_1/model_1.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_1/model_2.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_1/model_3.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_1/model_4.pth",
                ],
                2: [
                    "result_risk/ETHUSDT/potential_model/initial_action_2/model_0.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_2/model_1.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_2/model_2.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_2/model_3.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_2/model_4.pth",
                ],
                3: [
                    "result_risk/ETHUSDT/potential_model/initial_action_3/model_0.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_3/model_1.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_3/model_2.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_3/model_3.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_3/model_4.pth",
                ],
                4: [
                    "result_risk/ETHUSDT/potential_model/initial_action_4/model_0.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_4/model_1.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_4/model_2.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_4/model_3.pth",
                    "result_risk/ETHUSDT/potential_model/initial_action_4/model_4.pth",
                ],
            }
        elif args.dataset_name == "GALAUSDT":
            model_path_list_dict = {
                0: [
                    "result_risk/GALAUSDT/potential_model/initial_action_0/model_0.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_0/model_1.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_0/model_2.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_0/model_3.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_0/model_4.pth",
                ],
                1: [
                    "result_risk/GALAUSDT/potential_model/initial_action_1/model_0.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_1/model_1.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_1/model_2.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_1/model_3.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_1/model_4.pth",
                ],
                2: [
                    "result_risk/GALAUSDT/potential_model/initial_action_2/model_0.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_2/model_1.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_2/model_2.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_2/model_3.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_2/model_4.pth",
                ],
                3: [
                    "result_risk/GALAUSDT/potential_model/initial_action_3/model_0.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_3/model_1.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_3/model_2.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_3/model_3.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_3/model_4.pth",
                ],
                4: [
                    "result_risk/GALAUSDT/potential_model/initial_action_4/model_0.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_4/model_1.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_4/model_2.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_4/model_3.pth",
                    "result_risk/GALAUSDT/potential_model/initial_action_4/model_4.pth",
                ],
            }

        self.model_path_list_dict = model_path_list_dict
        self.num_model = len(self.model_path_list_dict[0])
        self.max_holding_number = args.max_holding_number
        self.action_dim = args.action_dim
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.reward_scale = args.reward_scale

        # RL setting

        self.train_data_path = args.train_data_path
        self.high_level_tech_indicator_list = np.load(
            "data/feature/minitue_feature.npy"
        ).tolist()
        self.low_level_tech_indicator_list = np.load(
            "data/feature/second_feature.npy"
        ).tolist()
        self.n_state = len(self.high_level_tech_indicator_list)
        self.update_counter = 0
        self.grad_clip = 10
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.update_times = args.update_times
        self.gamma = args.gamma
        self.epsilon_init = args.epsilon_init
        self.epsilon_min = args.epsilon_min
        self.epsilon_step = args.epsilon_step
        self.epsilon_decay = (self.epsilon_init - self.epsilon_min) / self.epsilon_step
        self.epsilon = self.epsilon_init
        self.target_freq = args.target_freq
        # replay buffer setting
        self.n_step = args.n_step
        self.buffer_size = args.buffer_size
        # general learning setting
        self.lr_init = args.lr_init
        self.lr_min = args.lr_min
        self.lr_step = args.lr_step
        self.lr_decay = (self.lr_init - self.lr_min) / self.lr_step
        self.lr = self.lr_init
        self.num_sample = args.num_sample
        # network
        self.hidden_nodes = args.hidden_nodes
        self.eval_net = Qnet_high_level(
            self.n_state, self.num_model, self.hidden_nodes
        ).to(self.device)
        self.target_net = copy.deepcopy(self.eval_net)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        

    def update(
        self,
        info: dict,
        actions: torch.tensor,
        rewards: torch.tensor,
        info_: dict,
        dones: torch.tensor,
    ):
        # TD error
        q_eval = self.eval_net(
            torch.squeeze(info["high_level_state"]),
            
        ).gather(1, actions)
        q_next = self.target_net(
            torch.squeeze(info_["high_level_state"]),
        ).detach()
        # since investigating is a open end problem, we do not use the done here to update
        q_target = rewards + torch.max(q_next, 1)[0].view(self.batch_size, 1) * (
            1 - dones
        )*self.gamma

        td_error = self.loss_func(q_eval, q_target)

        # final loss function
        loss = td_error
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.grad_clip)
        self.optimizer.step()
        for param, target_param in zip(
            self.eval_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        self.update_counter += 1

        return (
            td_error.cpu(),
            torch.mean(q_eval.cpu()),
            torch.mean(q_target.cpu()),
            torch.mean(rewards.cpu()),
            torch.std(rewards.cpu()),
        )

    def act(self, info, epsilon):
        x = torch.tensor(info["high_level_state"]).to(self.device).to(torch.float32)
        

        if np.random.uniform() > epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = random.choice(range(self.num_model))
        return action

    def act_test(self, info):
        x = torch.tensor(info["high_level_state"]).to(self.device).to(torch.float32)
        
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        # epoch_number = int(len(self.train_df) / self.chunk_length)
        epoch_number = 1
        return_rate_list = []
        replay_buffer = Multi_step_ReplayBuffer_multi_info(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        step_counter = 0
        for sample in range(self.num_sample):
            self.train_df = pd.read_feather(self.train_data_path)

            train_env = high_level_testing_env(
                df=self.train_df,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                action_dim=self.action_dim,
                early_stop=0,
                initial_action=0,
                model_path_list_dict=self.model_path_list_dict,
            )
            s, info = train_env.reset()
            episode_reward_sum = 0
            while True:
                a = self.act(info, self.epsilon)
                self.epsilon = (
                    self.epsilon - self.epsilon_decay
                    if self.epsilon - self.epsilon_decay > self.epsilon_min
                    else self.epsilon_min
                )
                self.lr = (
                    self.lr - self.lr_decay
                    if self.lr - self.lr_decay > self.lr_min
                    else self.lr_min
                )
                for p in self.optimizer.param_groups:
                    p["lr"] = self.lr

                s_, r, done, info_ = train_env.step(a)
                if not done:
                    replay_buffer.add(s, info, a, r, s_, info_, done)
                episode_reward_sum += r

                s, info = s_, info_
                step_counter += 1
                if (
                    step_counter > (self.batch_size + self.n_step)
                    and step_counter % self.target_freq == 1
                ):
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
                            infos,
                            actions,
                            rewards,
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
                # self.test(epoch_path)
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_required_money_train_list = []
                epoch_reward_sum_train_list = []

    def test(self, epoch_path):
        self.eval_net.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl"))
        )
        for name, df in zip(
            ["valid", "test"],
            self.test_df_list,
        ):
            self.test_df = df
            self.test_ev_instance = high_level_testing_env(
                df=self.test_df,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                action_dim=self.action_dim,
                early_stop=0,
                initial_action=0,
                model_path_list_dict=self.model_path_list_dict,
            )
            s, info = self.test_ev_instance.reset()
            done = False
            action_list = []
            reward_list = []
            while not done:
                a = self.act_test(info)
                s_, r, done, info_ = self.test_ev_instance.step(a)
                reward_list.append(r)
                s = s_
                info = info_
                action_list.append(a)
            (
                portfit_magine,
                final_balance,
                required_money,
                commission_fee,
            ) = self.test_ev_instance.get_final_return_rate(slient=True)

            action_list = np.array(action_list)
            reward_list = np.array(reward_list)
            if not os.path.exists(os.path.join(epoch_path, name)):
                os.makedirs(os.path.join(epoch_path, name))
            np.save(os.path.join(epoch_path, name, "action.npy"), action_list)
            np.save(os.path.join(epoch_path, name, "reward.npy"), reward_list)
            np.save(
                os.path.join(epoch_path, name, "final_balance.npy"),
                self.test_ev_instance.final_balance,
            )
            np.save(
                os.path.join(epoch_path, name, "pure_balance.npy"),
                self.test_ev_instance.pured_balance,
            )
            np.save(
                os.path.join(epoch_path, name, "require_money.npy"),
                self.test_ev_instance.required_money,
            )
            np.save(
                os.path.join(epoch_path, name, "commission_fee_history.npy"),
                self.test_ev_instance.comission_fee_history,
            )
            get_test_contrast_curve_high_level(
                self.test_df,
                os.path.join(epoch_path, name, "result.pdf"),
                reward_list,
                self.test_ev_instance.required_money,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    agent = DQN(args)
    agent.train()
