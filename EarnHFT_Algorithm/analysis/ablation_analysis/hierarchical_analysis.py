import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re


def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)


def get_single_agent_contrast(
    reward_list_list, require_money_list, algorithm_names, colors, save_path
):
    fig = plt.figure(figsize=(7.5, 5))
    for reward_list, require_money, algorithm_name, color in zip(
        reward_list_list, require_money_list, algorithm_names, colors
    ):
        accummulative_reward_sum = [reward_list[0]]
        for i in range(len(reward_list) - 1):
            accummulative_reward_sum.append(
                accummulative_reward_sum[-1] + reward_list[i + 1]
            )
        our_net_curve = np.array(accummulative_reward_sum) / (require_money + 1e-12)

        plt.plot(
            range(len(our_net_curve)),
            our_net_curve * 100,
            color=color,
            label=algorithm_name,
        )
    plt.xlabel("Trading Timestamp(s)", size=18)
    plt.ylabel("Total Return(%)", size=18)
    plt.grid(ls="--")
    plt.legend(
        loc="upper center",
        fancybox=True,
        ncol=6,
        fontsize=8,
        columnspacing=2,
        bbox_to_anchor=(0.45, 1.1, 0, 0),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "hirerchial_abligation.pdf"))
    plt.close()


if __name__ == "__main__":
    
    
    # save_path = "result_risk/BTCTUSD/high_level_single_agent"
    # best_path = "result_risk/BTCTUSD/high_level/seed_12345/epoch_54"
    
    
    
    # save_path = "result_risk/BTCUSDT/high_level_single_agent"
    # best_path = "result_risk/BTCUSDT/high_level/seed_12345/epoch_42"
    
    
    # save_path = "result_risk/ETHUSDT/high_level_single_agent"
    # best_path = "result_risk/ETHUSDT/high_level/seed_12345/epoch_58"
    
    
    save_path = "result_risk/GALAUSDT/high_level_single_agent"
    best_path = "result_risk/GALAUSDT/high_level/seed_12345/epoch_27"
    
    algorithm_names = [
        "dynamics_1",
        "dynamics_2",
        "dynamics_3",
        "dynamics_4",
        "dynamics_5",
        "high_level",
    ]
    colors = [
        "moccasin",
        "aquamarine",
        "#dbc2ec",
        "orchid",
        "lightskyblue",
        "lightslategrey",
        "orange",
        "lightcoral",
    ]
    agent_list = os.listdir(save_path)
    reward_list_list = []
    require_money_list = []
    sort_list(agent_list)

    for agent in agent_list:
        reward_list = np.load(
            os.path.join(save_path, agent, "test", "macro_reward_history.npy")
        )
        require_money = np.load(
            os.path.join(save_path, agent, "test", "require_money.npy")
        )
        reward_list_list.append(reward_list)
        require_money_list.append(require_money)
    reward_list = np.load(os.path.join(best_path, "test", "macro_reward_history.npy"))
    require_money = np.load(
        os.path.join(best_path, "test", "require_money.npy")
    )
    reward_list_list.append(reward_list)
    require_money_list.append(require_money)
    get_single_agent_contrast(
        reward_list_list, require_money_list, algorithm_names, colors, save_path
    )
