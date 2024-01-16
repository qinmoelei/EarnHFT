import torch
import sys
import random
import os

sys.path.append(".")
import re
from model.net import *
from torch.utils.tensorboard import SummaryWriter
from RL.agent.base.util.replay_buffer_PPO import ReplayBuffer_lstm
from env.low_level_env import Testing_env
import argparse

parser = argparse.ArgumentParser()
# basic setting
parser.add_argument(
    "--result_path",
    type=str,
    default="result_risk",
    help="the path for storing the test result",
)
# PPO setting
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="the path for storing the test result",
)


parser.add_argument(
    "--mini_batch_size",
    type=int,
    default=64,
    help="the path for storing the test result",
)



# parser.add_argument(
#     "--batch_size",
#     type=int,
#     default=4,
#     help="the path for storing the test result",
# )


# parser.add_argument(
#     "--mini_batch_size",
#     type=int,
#     default=2,
#     help="the path for storing the test result",
# )
# network
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="The number of neurons in hidden layers of the neural network",
)
parser.add_argument("--lr_init", type=float, default=5e-5, help="the learning rate")
parser.add_argument("--lr_min", type=float, default=1e-5, help="the learning rate")
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
    default=30,
    help="the scale factor we put in reward",
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


class PPO_discrete_RNN:
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # log path
        # self.model_path = os.path.join(
        #     args.result_path,
        #     args.dataset_name,
        #     "dra",
        #     "seed_{}".format(self.seed),
        # )
        self.model_path = os.path.join(
            args.result_path,
            args.dataset_name,
            "dra_short",
            "seed_{}".format(self.seed),
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        # env setting
        self.max_holding_number = args.max_holding_number
        self.action_dim = args.action_dim
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.reward_scale = args.reward_scale
        # training setting
        self.num_sample = args.num_sample
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        # Learning rate of actor and critic
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

        # data
        self.train_data_path = args.train_data_path

        # self.episode_limit = 3600
        self.episode_limit = 120
        self.chunk_num = 14400
        # network
        self.tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()
        self.hidden_nodes = args.hidden_nodes
        self.n_state = len(self.tech_indicator_list)
        self.ac = Actor_Critic_RNN(
            state_dim=self.n_state,
            hidden_dim=self.hidden_nodes,
            action_dim=self.action_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)
        # replay buffer
        self.replay_buffer = ReplayBuffer_lstm(
            self.gamma,
            self.lamda,
            self.n_state,
            self.action_dim,
            self.episode_limit,
            self.batch_size,
        )

    def reset_rnn_hidden(self):
        self.ac.actor_rnn_hidden = None
        self.ac.critic_rnn_hidden = None

    def choose_action(self, s, info, evaluate=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor([info["previous_action"]]).float().to(self.device), 0
            ).to(self.device)
            avaliable_action = torch.unsqueeze(
                torch.tensor(info["avaliable_action"]).to(self.device), 0
            ).to(self.device)


            logit = self.ac.actor(s, previous_action, avaliable_action)
            if evaluate:
                a = torch.argmax(logit)
                return a.item(), None
            else:
                dist = Categorical(logits=logit)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
                return a.item(), a_logprob.item()

    def get_value(self, s, info):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor([info["previous_action"]]).float().to(self.device), 0
            ).to(self.device)
            value = self.ac.critic(s, previous_action)
            return value.item()

    def update(self, replay_buffer: ReplayBuffer_lstm):
        batch = replay_buffer.get_training_data()  # Get training data
        print("update")
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SequentialSampler(range(self.batch_size)), self.mini_batch_size, False
            ):
                # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                self.reset_rnn_hidden()

                logits_now = self.ac.actor(
                    batch["s"][index].to(self.device),
                    batch["p_a"][index].to(self.device),
                    batch["a_a"][index].to(self.device),
                )  # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                values_now = self.ac.critic(
                    batch["s"][index].to(self.device),
                    batch["p_a"][index].to(self.device),
                ).squeeze(
                    -1
                )  # values_now.shape=(mini_batch_size, max_episode_len)

                dist_now = Categorical(logits=logits_now.to(self.device))
                dist_entropy = (dist_now.entropy()).to(
                    self.device
                )  # shape(mini_batch_size, max_episode_len)
                a_logprob_now = dist_now.log_prob(batch["a"][index].to(self.device)).to(
                    self.device
                )  # shape(mini_batch_size, max_episode_len)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(
                    a_logprob_now.to(self.device)
                    - batch["a_logprob"][index].to(self.device)
                ).to(
                    self.device
                )  # shape(mini_batch_size, max_episode_len)

                # actor loss
                surr1 = ratios * batch["adv"][index].to(self.device)
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch[
                    "adv"
                ][index].to(self.device)
                actor_loss = (
                    -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                ).to(
                    self.device
                )  # shape(mini_batch_size, max_episode_len)
                actor_loss = (
                    actor_loss * batch["active"][index].to(self.device)
                ).sum() / batch["active"][index].sum().to(self.device)

                # critic_loss
                critic_loss = (
                    values_now - batch["v_target"][index].to(self.device)
                ) ** 2
                critic_loss = (
                    critic_loss * batch["active"][index].to(self.device)
                ).sum() / batch["active"][index].to(self.device).sum()

                # Update
                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss * 0.5
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

    def save_model(self, epoch_path):
        torch.save(
            self.ac.state_dict(),
            os.path.join(epoch_path, "trained_model.pkl"),
        )

    def load_model(self, epoch_path):
        self.ac.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl"))
        )

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
                self.reset_rnn_hidden()
                for episode_step in range(self.episode_limit):
                    self.lr = (
                        self.lr - self.lr_decay
                        if self.lr - self.lr_decay > self.lr_min
                        else self.lr_min
                    )
                    for p in self.optimizer.param_groups:
                        p["lr"] = self.lr

                    a, a_logprob = self.choose_action(s, info, evaluate=False)
                    v = self.get_value(s, info)
                    s_, r, done, info_ = train_env.step(a)
                    episode_reward_sum += r

                    dw = False
                    self.replay_buffer.store_transition(
                        episode_step,
                        s,
                        info["previous_action"],
                        info["avaliable_action"],
                        v,
                        a,
                        a_logprob,
                        r*self.reward_scale,
                        dw,
                    )
                    s = s_
                    info = info_
                    if done:
                        break
                if not done:
                    v = self.get_value(s, info)
                    self.replay_buffer.store_last_value(episode_step + 1, v)

                    if self.replay_buffer.episode_num == self.batch_size:
                        self.update(self.replay_buffer)  # Training
                        self.replay_buffer.reset_buffer()
                        self.reset_rnn_hidden()
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
                    self.ac.state_dict(),
                    os.path.join(epoch_path, "trained_model.pkl"),
                )
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_required_money_train_list = []
                epoch_reward_sum_train_list = []
            
            
            

        


if __name__ == "__main__":
    args = parser.parse_args()
    agent = PPO_discrete_RNN(args)
    agent.train()
