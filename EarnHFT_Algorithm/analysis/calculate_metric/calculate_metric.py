import pandas as pd
import numpy as np
import os
import sys
from collections import OrderedDict

sys.path.append(".")
from analysis.calculate_metric.calculate_wsrc import print_metrics, evaluate_mintue, evaluate_second
import re


def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)


def pick_agent_and_generate_result_table(root_path, name):
    # pick agent from path
    if name != "rule_base":
        valid_result_list = []
        epoch_path_list = os.listdir(root_path)
        sort_list(epoch_path_list)
        epoch_path_list.remove("log")
        for epoch in epoch_path_list:
            epoch_path = os.path.join(root_path, epoch)
            valid_path = os.path.join(epoch_path, "valid")
            valid_result = np.load(os.path.join(valid_path, "final_balance.npy")) / (
                np.load(os.path.join(valid_path, "require_money.npy")) + 1e-12
            )
            valid_result_list.append(valid_result)
        valid_index = valid_result_list.index(max(valid_result_list))
        test_path = os.path.join(root_path, epoch_path_list[valid_index], "test")
        reward_list = np.load(os.path.join(test_path, "reward.npy"))
        require_money = np.load(os.path.join(test_path, "require_money.npy"))
        (
            tr,
            daily_vol,
            mdd,
            daily_dd,
            annual_sr,
            daily_cr,
            daily_SoR,
            annual_cr,
            annual_SoR,
            annual_vol,
        ) = evaluate_second(require_money, reward_list)
        stats = OrderedDict(
            {
                "Total Return": ["{:04f}%".format(tr * 100)],
                # "Daily Volatility": ["{:04f}%".format(daily_vol * 100)],
                "Annualized Sharp Ratio": ["{:04f}".format(annual_sr)],
                "Annualized Calmar Ratio": ["{:04f}".format(annual_cr)],
                "Annualized Sortino Ratio": ["{:04f}".format(annual_SoR)],
                "Annualized Volatility": ["{:04f}%".format(annual_vol * 100)],
                "Annualized Downside Deviation": ["{:04f}%".format(daily_dd * 100*np.sqrt(365))],
                "Max Drawdown": ["{:04f}%".format(mdd * 100)],
                # "Daily Calmar Ratio": ["{:04f}".format(daily_cr)],
                # "Daily Sortino Ratio": ["{:04f}".format(daily_SoR)],
            }
        )
        table = print_metrics(stats)
        print(name)
        print(epoch_path_list[valid_index])
        print(table)

    else:
        valid_result_list = []
        different_para = os.listdir(root_path)
        for par in different_para:
            par_path = os.path.join(root_path, par)
            test_path = os.path.join(par_path, "test")
            reward_list = np.load(os.path.join(test_path, "reward.npy"))
            require_money = np.load(os.path.join(test_path, "require_money.npy"))
            (
                tr,
                daily_vol,
                mdd,
                daily_dd,
                annual_sr,
                daily_cr,
                daily_SoR,
                annual_cr,
                annual_SoR,
                annual_vol,
            ) = evaluate_second(require_money, reward_list)
            stats = OrderedDict(
                {
                    "Total Return": ["{:04f}%".format(tr * 100)],
                    # "Daily Volatility": ["{:04f}%".format(daily_vol * 100)],
                    "Annualized Sharp Ratio": ["{:04f}".format(annual_sr)],
                    "Annualized Calmar Ratio": ["{:04f}".format(annual_cr)],
                    "Annualized Sortino Ratio": ["{:04f}".format(annual_SoR)],
                    "Annualized Volatility": ["{:04f}%".format(annual_vol * 100)],
                    "Annualized Downside Deviation": ["{:04f}%".format(daily_dd * 100*np.sqrt(365))],
                    "Max Drawdown": ["{:04f}%".format(mdd * 100)],
                    # "Daily Calmar Ratio": ["{:04f}".format(daily_cr)],
                    # "Daily Sortino Ratio": ["{:04f}".format(daily_SoR)],
                }
            )
            table = print_metrics(stats)
            print(name)
            print(par)
            print(table)


if __name__ == "__main__":
    root_path_list = [
        "result_risk/GALAUSDT/cdqn_rp/seed_12345",
        "result_risk/GALAUSDT/dqn_ada_0.0/seed_12345",
        "result_risk/GALAUSDT/dra_short/seed_12345",
        "result_risk/GALAUSDT/ppo/seed_12345",
        "result_risk/GALAUSDT/rule_base",
    ]
    names = ["cdqn_rp", "dqn", "dra_short", "ppo", "rule_base"]
    for root_path, name in zip(root_path_list, names):
        print(root_path)
        pick_agent_and_generate_result_table(root_path, name)
