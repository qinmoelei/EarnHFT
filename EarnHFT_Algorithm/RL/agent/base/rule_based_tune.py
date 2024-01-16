import pandas as pd
import numpy as np
import sys

sys.path.append(".")
from env.low_level_env import Testing_env
import os
import random
from tqdm import tqdm
from RL.util.graph import get_test_contrast_curve

# to compute a rule based method in HFT, we need 3 things: the trading signal, stop win and stop loss.
# Here we are going to use MACD and Imbalance volume as the trading signal and tunning the stop win and stop lose
# Dual-component model of respiratory motion based on the periodic ARMA method stands for the mathmatical method (Pure arma model with AIC to test p and q)
# there will only be 2 kinds of position, all in and all out representing previous action 4 and 0 respectively
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--technicail_indicator",
    type=str,
    default="MACD",
    choices=["MACD", "Imbalance_Volume",],
    help="the indicator we use to open or close a position",
)


parser.add_argument(
    "--stop_win",
    type=float,
    default=0.1,
    help="the stop win for trading",
)
parser.add_argument(
    "--stop_lose",
    type=float,
    default=0.1,
    help="the stop lose for trading",
)


parser.add_argument(
    "--long_term",
    type=int,
    default=26,
    help="the stop win for trading",
)

parser.add_argument(
    "--mid_term",
    type=int,
    default=12,
    help="the stop lose for trading",
)


parser.add_argument(
    "--short_term",
    type=int,
    default=9,
    help="the stop lose for trading",
)








parser.add_argument(
    "--upper_theshold",
    type=float,
    default=0.1,
    help="the stop lose for trading",
)


parser.add_argument(
    "--lower_theshold",
    type=float,
    default=-0.1,
    help="the stop lose for trading",
)



parser.add_argument(
    "--result_path",
    type=str,
    default="result_risk",
    help="the path of test model",
)
parser.add_argument(
    "--valid_data_path",
    type=str,
    default="data/BTCTUSD/valid.feather",
    help="training data chunk",
)

parser.add_argument(
    "--test_data_path",
    type=str,
    default="data/BTCTUSD/test.feather",
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
parser.add_argument(
    "--action_dim",
    type=int,
    default=5,
    help="the number of action we have in the training and testing env",
)


class rule_base_trader:
    def __init__(self, args):
        if args.technicail_indicator=="MACD":
            self.model_path = os.path.join(
                args.result_path,
                args.dataset_name,
                "rule_base",
                "{}_{}_{}_{}".format(
                    args.technicail_indicator,
                    args.long_term,
                    args.mid_term,
                    args.short_term,
                ),
            )
        if args.technicail_indicator=="Imbalance_Volume":
            self.model_path = os.path.join(
                args.result_path,
                args.dataset_name,
                "rule_base",
                "{}_{}_{}".format(
                    args.technicail_indicator,
                    args.upper_theshold,
                    args.lower_theshold,
                ),
            )
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.valid_data_path = args.valid_data_path
        self.test_data_path = args.test_data_path

        self.max_holding_number = args.max_holding_number
        self.transcation_cost = args.transcation_cost
        self.action_dim = args.action_dim
        self.technicail_indicator = args.technicail_indicator
        self.stop_lose = args.stop_lose
        self.stop_win = args.stop_win
        self.long_term = args.long_term
        self.mid_term = args.mid_term
        self.short_term = args.short_term
        self.upper_theshold=args.upper_theshold
        self.lower_theshold=args.lower_theshold

    def test(self):
        for name, data_path in zip(
            ["valid", "test"], [self.valid_data_path, self.test_data_path]
        ):
            true_action_list = []
            reward_list = []
            self.tech_indicator_list = np.load(
                "data/feature/second_feature.npy"
            ).tolist()
            self.test_df = pd.read_feather(data_path)
            import_fea = "bid1_price"
            # generate action based on the indicator
            # the tuining part, we use the different paremater to do the trick
            if self.technicail_indicator == "MACD":
                df_singnal = pd.DataFrame()
                DIF = (
                    self.test_df[import_fea].ewm(span=self.mid_term).mean()
                    - self.test_df[import_fea].ewm(span=self.long_term).mean()
                )
                df_singnal["DIF"] = DIF
                DEA = df_singnal["DIF"].ewm(span=self.short_term).mean()
                df_singnal["DEA"] = DEA
                MACD = DIF - DEA
                df_singnal["MACD"] = MACD
                # when MACD is positive, we buy and hold, when MACD is nagetive, we sell and hold
                # More precisely DIF >0 and MACD>0 buy, DIF>0，MACD<0 or DIF<0,MACD>0 we hold because there is where the
                # trend is not clear, when DIF<0,MACD<0, we sell.
                action_list = []
                for macd, dif in zip(MACD, DIF):
                    if macd > 0 and dif > 0:
                        action_list.append("buy")
                    elif macd * dif <= 0:
                        action_list.append("hold")
                    else:
                        action_list.append("sell")

            if self.technicail_indicator == "Imbalance_Volume":
                
                Imbalance_Volume=self.test_df["imblance_volume_oe"]
                # when MACD is positive, we buy and hold, when MACD is nagetive, we sell and hold
                # More precisely DIF >0 and MACD>0 buy, DIF>0，MACD<0 or DIF<0,MACD>0 we hold because there is where the
                # trend is not clear, when DIF<0,MACD<0, we sell.
                action_list = []
                for imbalance_volume in Imbalance_Volume:
                    if imbalance_volume>=self.upper_theshold:
                        action_list.append("buy")
                    elif imbalance_volume<self.upper_theshold and imbalance_volume>self.lower_theshold:
                        action_list.append("hold")
                    else:
                        action_list.append("sell")






            self.test_env_instance = Testing_env(
                df=self.test_df,
                tech_indicator_list=self.tech_indicator_list,
                transcation_cost=self.transcation_cost,
                back_time_length=1,
                max_holding_number=self.max_holding_number,
                action_dim=self.action_dim,
                initial_action=0,
            )
            s, info = self.test_env_instance.reset()
            previous_true_action = 0
            for action in action_list:
                avaliable_index = [
                    index
                    for index, value in enumerate(info["avaliable_action"])
                    if value == 1
                ]
                if action == "buy":
                    true_action = max(avaliable_index)
                if action == "hold":
                    true_action = previous_true_action
                if action == "sell":
                    true_action = min(avaliable_index)
                s_, r, done, info_ = self.test_env_instance.step(true_action)
                reward_list.append(r)
                true_action_list.append(true_action)
                s = s_
                info = info_
                previous_true_action = true_action
                if done:
                    break
            if not os.path.exists(os.path.join(self.model_path, name)):
                os.makedirs(os.path.join(self.model_path, name))
            np.save(os.path.join(self.model_path, name, "action.npy"), true_action_list)
            np.save(os.path.join(self.model_path, name, "reward.npy"), reward_list)
            np.save(
                os.path.join(self.model_path, name, "final_balance.npy"),
                self.test_env_instance.final_balance,
            )
            np.save(
                os.path.join(self.model_path, name, "pure_balance.npy"),
                self.test_env_instance.pured_balance,
            )
            np.save(
                os.path.join(self.model_path, name, "require_money.npy"),
                self.test_env_instance.required_money,
            )
            np.save(
                os.path.join(self.model_path, name, "commission_fee_history.npy"),
                self.test_env_instance.comission_fee_history,
            )

            get_test_contrast_curve(
                self.test_df,
                os.path.join(self.model_path, name, "result.pdf"),
                reward_list,
                self.test_env_instance.required_money,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    agent = rule_base_trader(args)
    agent.test()
