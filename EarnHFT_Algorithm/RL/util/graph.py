import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_sample_curve(df: pd.DataFrame, save_path):
    df = df.reset_index()
    buy_hold_df_net_curve = np.array(df.bid1_price / (df.ask1_price.iloc[0])) - 1
    plt.plot(
        range(len(buy_hold_df_net_curve)),
        buy_hold_df_net_curve * 100,
        color="coral",
        label="Buy & Hold",
    )
    plt.xlabel("Trading times", size=18)
    plt.ylabel("Total Return(%)", size=18)
    plt.grid(ls="--")
    plt.legend(
        loc="upper center",
        fancybox=True,
        ncol=2,
        fontsize="x-large",
        bbox_to_anchor=(0.49, 1.15, 0, 0),
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_test_contrast_curve(df: pd.DataFrame, save_path, reward_list, require_money):
    df = df.reset_index(drop=True)

    buy_hold_df_net_curve = np.array(df.bid1_price / (df.ask1_price.iloc[0])) - 1

    accummulative_reward_sum = [reward_list[0]]
    for i in range(len(reward_list) - 1):
        accummulative_reward_sum.append(
            accummulative_reward_sum[-1] + reward_list[i + 1]
        )
    our_net_curve = np.array(accummulative_reward_sum) / (require_money + 1e-12)

    plt.plot(
        range(len(buy_hold_df_net_curve)),
        buy_hold_df_net_curve * 100,
        color="coral",
        label="Buy & Hold",
    )

    plt.plot(
        range(len(our_net_curve)), our_net_curve * 100, color="royalblue", label="Ours"
    )
    plt.xlabel("Trading Timestamp(s)", size=18)
    plt.ylabel("Total Return(%)", size=18)
    plt.grid(ls="--")
    plt.legend(
        loc="upper center",
        fancybox=True,
        ncol=2,
        fontsize="x-large",
        bbox_to_anchor=(0.49, 1.15, 0, 0),
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_test_contrast_curve_high_level(
    df: pd.DataFrame, save_path, reward_list, require_money
):
    df = df[df["timestamp"].dt.second == 0].reset_index(drop=True)
    buy_hold_df_net_curve = np.array(df.bid1_price / (df.ask1_price.iloc[0])) - 1

    accummulative_reward_sum = [reward_list[0]]
    for i in range(len(reward_list) - 1):
        accummulative_reward_sum.append(
            accummulative_reward_sum[-1] + reward_list[i + 1]
        )
    our_net_curve = np.array(accummulative_reward_sum) / (require_money + 1e-12)

    plt.plot(
        range(len(buy_hold_df_net_curve)),
        buy_hold_df_net_curve * 100,
        color="coral",
        label="Buy & Hold",
    )

    plt.plot(
        range(len(our_net_curve)), our_net_curve * 100, color="royalblue", label="Ours"
    )
    plt.xlabel("Trading times", size=18)
    plt.ylabel("Total Return(%)", size=18)
    plt.grid(ls="--")
    plt.legend(
        loc="upper center",
        fancybox=True,
        ncol=2,
        fontsize="x-large",
        bbox_to_anchor=(0.49, 1.15, 0, 0),
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    df = pd.read_feather("data/train.feather")
    get_sample_curve(df, "data/train.pdf")
