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
from RL.util.graph import get_test_contrast_curve

parser = argparse.ArgumentParser()
# basic setting
parser.add_argument(
    "--test_path",
    type=str,
    default="result_risk/BTCTUSD/dra/seed_12345/epoch_1",
    help="the path of test model",
)

# network
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="The number of neurons in hidden layers of the neural network",
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
    "--dataset_name",
    type=str,
    default="BTCTUSD",
    help="the name of the dataset, used for log",
)
parser.add_argument(
    "--valid_data_path",
    type=str,
    default="data/BTCTUSD/valid.feather",
    help="the path stored for train df list",
)
parser.add_argument(
    "--test_data_path",
    type=str,
    default="data/BTCTUSD/test.feather",
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
        self.device = "cuda"

        # log path
        self.test_path = args.test_path

        # env setting
        self.max_holding_number = args.max_holding_number
        self.action_dim = args.action_dim
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        

        # data
        self.valid_data_path = args.valid_data_path
        self.test_data_path = args.test_data_path


        # network
        self.tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()
        self.hidden_nodes = args.hidden_nodes
        self.n_state = len(self.tech_indicator_list)
        self.ac = Actor_Critic_RNN(
            state_dim=self.n_state,
            hidden_dim=self.hidden_nodes,
            action_dim=self.action_dim,
        ).to(self.device)
        
    def reset_rnn_hidden(self):
        self.ac.actor_rnn_hidden = None
        self.ac.critic_rnn_hidden = None

    def choose_action(self, s, info, evaluate=True):
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

    

    def load_model(self, epoch_path):
        self.ac.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl"),map_location=torch.device(self.device))
        )

    def test(self):
        self.load_model(self.test_path)
        self.reset_rnn_hidden()
        for name,data_path in zip(["valid","test"],[self.valid_data_path,self.test_data_path]):
            action_list = []
            reward_list = []
            self.test_df=pd.read_feather(data_path)
            self.test_env_instance = Testing_env(
                        df=self.test_df,
                        tech_indicator_list=self.tech_indicator_list,
                        transcation_cost=self.transcation_cost,
                        back_time_length=self.back_time_length,
                        max_holding_number=self.max_holding_number,
                        action_dim=self.action_dim,
                        initial_action=0,
                    )
            s, info = self.test_env_instance.reset()
            index=0
            while True:
                if index%3600==0:
                    self.reset_rnn_hidden()
                a, a_logprob = self.choose_action(s, info, evaluate=True)
                s_, r, done, info_ = self.test_env_instance.step(a)
                action_list.append(a)
                reward_list.append(r)
                s = s_
                info = info_
                index+=1
                if done:
                    break
            if not os.path.exists(os.path.join(self.test_path, name)):
                os.makedirs(os.path.join(self.test_path, name))
            np.save(os.path.join(self.test_path, name, "action.npy"), action_list)
            np.save(os.path.join(self.test_path, name, "reward.npy"), reward_list)
            np.save(
                os.path.join(self.test_path, name, "final_balance.npy"),
                self.test_env_instance.final_balance,
            )
            np.save(
                os.path.join(self.test_path, name, "pure_balance.npy"),
                self.test_env_instance.pured_balance,
            )
            np.save(
                os.path.join(self.test_path, name, "require_money.npy"),
                self.test_env_instance.required_money,
            )
            np.save(
                os.path.join(self.test_path, name, "commission_fee_history.npy"),
                self.test_env_instance.comission_fee_history,
            )

            get_test_contrast_curve(
                self.test_df,
                os.path.join(self.test_path, name, "result.pdf"),
                reward_list,
                self.test_env_instance.required_money,
            )

            


if __name__ == "__main__":
    args = parser.parse_args()
    agent = PPO_discrete_RNN(args)
    agent.test()
