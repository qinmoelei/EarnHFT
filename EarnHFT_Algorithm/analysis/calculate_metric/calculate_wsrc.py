import pandas as pd
import numpy as np
import os
import re
from collections import OrderedDict
import prettytable
import matplotlib
import matplotlib.pyplot as plt


# calculate traditional finanace metric
def print_metrics(stats):
    table = prettytable.PrettyTable()
    # table.add_row(['' for _ in range(len(stats))])
    for key, value in stats.items():
        table.add_column(key, value)
    return table


def evaluate_mintue(require_money, reward_list):
    reward_sum_list = [reward_list[0]]
    for i in range(1, len(reward_list)):
        reward_sum_list.append(reward_sum_list[-1] + reward_list[i])
    total_asset_value_list = (require_money + np.array(reward_sum_list)).tolist()
    second_return_rate_list = [
        total_asset_value_list[i + 1] / total_asset_value_list[i] - 1
        for i in range(len(total_asset_value_list) - 1)
    ]
    tr = total_asset_value_list[-1] / total_asset_value_list[0] - 1
    vol = np.std(second_return_rate_list)
    daily_vol = vol * np.sqrt(60 * 24)
    mdd = 0
    peak = total_asset_value_list[0]
    for value in total_asset_value_list:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > mdd:
            mdd = dd
    negative_second_return_rate_list = [x for x in second_return_rate_list if x < 0]
    downside_deviation = np.std(negative_second_return_rate_list)
    daily_dd = downside_deviation * np.sqrt(60 * 24)
    sr = np.mean(second_return_rate_list) / np.std(second_return_rate_list)
    annual_sr = sr * np.sqrt(24 * 60 * 365)
    cr = np.mean(second_return_rate_list) / mdd
    daily_cr = cr * 24 * 60
    SoR = np.mean(second_return_rate_list) / downside_deviation
    daily_SoR = SoR * np.sqrt(60 * 24)
    return tr, daily_vol, mdd, daily_dd, annual_sr, daily_cr, daily_SoR


def evaluate_second(require_money, reward_list):
    reward_sum_list = [reward_list[0]]
    for i in range(1, len(reward_list)):
        reward_sum_list.append(reward_sum_list[-1] + reward_list[i])
    total_asset_value_list = (require_money + np.array(reward_sum_list)).tolist()
    second_return_rate_list = [
        total_asset_value_list[i + 1] / total_asset_value_list[i] - 1
        for i in range(len(total_asset_value_list) - 1)
    ]
    tr = total_asset_value_list[-1] / total_asset_value_list[0] - 1
    vol = np.std(second_return_rate_list)
    daily_vol = vol * np.sqrt(3600 * 24)
    annual_vol = vol * np.sqrt(3600 * 24 * 365)
    mdd = 0
    peak = total_asset_value_list[0]
    for value in total_asset_value_list:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > mdd:
            mdd = dd
    negative_second_return_rate_list = [x for x in second_return_rate_list if x < 0]
    downside_deviation = np.std(negative_second_return_rate_list)
    daily_dd = downside_deviation * np.sqrt(3600 * 24)
    sr = np.mean(second_return_rate_list) / np.std(second_return_rate_list)
    annual_sr = sr * np.sqrt(24 * 60 * 60 * 365)
    cr = np.mean(second_return_rate_list) / mdd
    daily_cr = cr * 24 * 60 * 60
    annual_cr = daily_cr * 365
    SoR = np.mean(second_return_rate_list) / downside_deviation
    daily_SoR = SoR * np.sqrt(60 * 24 * 60)
    annual_SoR = daily_SoR * np.sqrt(365)
    return (
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
    )


def calculate_portion_model(model_history: list, path):
    prob_list = []
    total_sample_number = len(model_history)
    model_history = model_history.tolist()
    for i in range(25):
        prob = model_history.count(i) / total_sample_number
        prob_list.append(prob)
    xticks = np.arange(25)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.yaxis.grid(linestyle="--", color="lightgray")
    # 现在先画
    print(prob_list)
    ax1.bar(
        xticks - 0.25,
        np.array(prob_list) * 100,
        color=[
            "moccasin",
            "aquamarine",
            "#dbc2ec",
            "orchid",
            "lightskyblue",
            "lightslategrey",
            "orange",
            "lightcoral",
            "moccasin",
            "aquamarine",
            "#dbc2ec",
            "orchid",
            "lightskyblue",
            "lightslategrey",
            "orange",
            "lightcoral",
            "moccasin",
            "aquamarine",
            "#dbc2ec",
            "orchid",
            "lightskyblue",
            "lightslategrey",
            "orange",
            "lightcoral",
            "moccasin",
        ],
        width=0.5,
    )
    ax1.set_ylabel("Probability(%)", fontsize=15)
    plt.savefig(os.path.join(path, "probability.pdf"), bbox_inches="tight")








if __name__ == "__main__":
    # root_path = "result_risk/BTCTUSD/high_level/seed_12345/epoch_54/test"
    # root_path = "result_risk/ETHUSDT/high_level/seed_12345/epoch_58/test"
    root_path = "result_risk/GALAUSDT/high_level/seed_12345/epoch_27/test"
    # root_path = "result_risk/BTCUSDT/high_level/seed_12345/epoch_42/test"

    # table result
    require_money = np.load(os.path.join(root_path, "require_money.npy"))
    reward_list = np.load(os.path.join(root_path, "macro_reward_history.npy"))
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
    print(table)

    # # plot probability
    # model_history = np.load(os.path.join(root_path, "model_history.npy"))
    # calculate_portion_model(model_history, root_path)
