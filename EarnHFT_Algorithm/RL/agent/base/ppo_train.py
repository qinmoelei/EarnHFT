# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/4.PPO-discrete
import sys

sys.path.append(".")
from RL.util.graph import get_test_contrast_curve
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import copy
from env.low_level_env import Testing_env, Training_Env
import pandas as pd
import yaml
from torch import nn
import torch
import numpy as np
from model.net import *
import argparse
from tqdm import tqdm
import random
from RL.agent.base.util.replay_buffer_PPO import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
# basic setting
parser.add_argument(
    "--result_path",
    type=str,
    default="result_risk",
    help="the path for storing the test result",
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the path for storing the test result",
)
parser.add_argument(
    "--action_dim",
    type=int,
    default=5,
    help="the number of action we have in the training and testing env",
)

parser.add_argument(
    "--reward_scale",
    type=float,
    default=30,
    help="the scale factor we put in reward",
)

parser.add_argument(
    "--back_time_length", type=int, default=1, help="the length of the holding period"
)


parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCTUSD",
    help="the name of the dataset, used for log",
)
parser.add_argument(
    "--train_data_path",
    type=str,
    default="data/BTCTUSD/train",
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

parser.add_argument(
    "--batch_size",
    type=int,
    default=3600,
    help="the max holding number of the coin",
)

parser.add_argument(
    "--mini_batch_size",
    type=int,
    default=512,
    help="the max holding number of the coin",
)
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="The number of neurons in hidden layers of the neural network",
)
parser.add_argument("--lr_init", type=float, default=5e-5, help="the learning rate")
parser.add_argument("--lr_min", type=float, default=1e-8, help="the learning rate")
parser.add_argument("--lr_step", type=float, default=1e6, help="the learning rate")
parser.add_argument("--gamma", type=float, default=1, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=1, help="PPO parameter")
parser.add_argument(
    "--num_sample",
    type=int,
    default=200,
    help="the overall number of sampling",
)
parser.add_argument(
    "--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy"
)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PPO(object):
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
            "ppo",
            "seed_{}".format(self.seed),
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)

        # environment setting
        self.max_holding_number = args.max_holding_number
        self.action_dim = args.action_dim
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.reward_scale = args.reward_scale

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.lr_init = args.lr_init
        self.lr_min = args.lr_min
        self.lr_step = args.lr_step
        self.lr_decay = (self.lr_init - self.lr_min) / self.lr_step
        self.lr = self.lr_init
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient

        self.num_sample = args.num_sample
        # data
        self.train_data_path = args.train_data_path

        self.episode_limit = 3600
        self.chunk_num = 14400
        # network
        self.tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()
        self.hidden_nodes = args.hidden_nodes
        self.n_state = len(self.tech_indicator_list)
        self.actor = Actor(
            N_STATES=self.n_state,
            hidden_nodes=self.hidden_nodes,
            N_ACTIONS=self.action_dim,
        ).to(self.device)
        self.critic = Critic(
            N_STATES=self.n_state,
            hidden_nodes=self.hidden_nodes,
        ).to(self.device)
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=1e-5
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr, eps=1e-5
        )

        self.replay_buffer = ReplayBuffer(
            batch_size=self.batch_size,
            state_dim=self.n_state,
            action_dim=self.action_dim,
            device=self.device,
        )

    def choose_action(self, s, info, evaluate=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor([info["previous_action"]]).float().to(self.device), 0
            ).to(self.device)
            avaliable_action = torch.unsqueeze(
                torch.tensor(info["avaliable_action"]).to(self.device), 0
            ).to(self.device)

            logit = self.actor(s, previous_action, avaliable_action)
            if evaluate:
                a = torch.argmax(logit)
                return a.item(), None
            else:
                dist = Categorical(logits=logit)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
                return a.item(), a_logprob.item()

    def update(self, replay_buffer: ReplayBuffer):
        print("updating")
        (
            s,
            p_a,
            a_a,
            a,
            a_logprob,
            r,
            s_,
            p_a_,
            dw,
            done,
        ) = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s, p_a)
            vs_ = self.critic(s_, p_a_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(
                reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())
            ):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)

            v_target = adv + vs
            
            
            # adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False
            ):
                dist_now = Categorical(
                    logits=self.actor(s[index], p_a[index], a_a[index])
                )
                dist_entropy = dist_now.entropy().view(
                    -1, 1
                )  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(
                    -1, 1
                )  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(
                    a_logprob_now - a_logprob[index]
                )  # shape(mini_batch_size X 1)

                surr1 = (
                    ratios * adv[index]
                )  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = (
                    torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                )
                actor_loss = (
                    -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                )  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index], p_a[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        df_number = len(os.listdir(self.train_data_path)) - 1
        epoch_number = 4
        random_position_list = random.choices(range(self.action_dim), k=self.num_sample)
        random_start_list = random.choices(range(df_number), k=self.num_sample)
        for sample in range(self.num_sample):
            df_index = random_start_list[sample]
            print("we are training with ", df_index)
            self.train_df = pd.read_feather(
                os.path.join(self.train_data_path, "df_{}.feather".format(df_index))
            )
            train_env = Testing_env(
                df=self.train_df,
                tech_indicator_list=self.tech_indicator_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                action_dim=self.action_dim,
                early_stop=0,
                initial_action=random_position_list[sample],
            )
            s, info = train_env.reset()
            episode_reward_sum = 0
            while True:
                self.lr = (
                    self.lr - self.lr_decay
                    if self.lr - self.lr_decay > self.lr_min
                    else self.lr_min
                )
                for p in self.optimizer_actor.param_groups:
                    p["lr"] = self.lr
                for p in self.optimizer_critic.param_groups:
                    p["lr"] = self.lr
                a, a_logprob = self.choose_action(s, info, evaluate=False)
                s_, r, done, info_ = train_env.step(a)
                episode_reward_sum += r

                dw = False
                self.replay_buffer.store(
                    s,
                    info["previous_action"],
                    info["avaliable_action"],
                    a,
                    a_logprob,
                    r * self.reward_scale,
                    s_,
                    info_["previous_action"],
                    dw,
                    done,
                )
                if self.replay_buffer.count == self.batch_size:
                    agent.update(self.replay_buffer)
                    self.replay_buffer.count = 0
                s = s_
                info = info_
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
                    self.actor.state_dict(),
                    os.path.join(epoch_path, "trained_model.pkl"),
                )
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_required_money_train_list = []
                epoch_reward_sum_train_list = []


if __name__ == "__main__":
    args = parser.parse_args()
    agent = PPO(args)
    agent.train()
