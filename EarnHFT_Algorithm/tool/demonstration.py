import pandas as pd
import numpy as np
import yaml
import sys
import os


def making_multi_level_dp_demonstration(df: pd.DataFrame,
                                        num_action,
                                        max_holding,
                                        commission_fee=0.000175,
                                        max_punish=1e12):
    # sell_value 和 buy_value与env之间的区别：没有超量一说 一旦超量直接把value打下来
    action_list = []

    def sell_value(price_information, position):
        # use bid price and size to evaluate
        value = 0
        # position 表示剩余的单量
        for i in range(1, 6):
            if position < price_information["bid{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["bid{}_size".format(i)]
                value += price_information["bid{}_price".format(
                    i)] * price_information["bid{}_size".format(i)]
        if position > 0 and i == 5:
            value = value - max_punish
            # 执行的单量
        else:
            value += price_information["bid{}_price".format(i)] * position
        # 卖的时候的手续费相当于少卖钱了

        return value * (1 - commission_fee)

    def buy_value(price_information, position):
        # this value measure how much
        value = 0
        for i in range(1, 6):
            if position < price_information["ask{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["ask{}_size".format(i)]
                value += price_information["ask{}_price".format(
                    i)] * price_information["ask{}_size".format(i)]
        if i == 5 and position > 0:
            value = value + max_punish
        else:
            value += price_information["ask{}_price".format(i)] * position
        # 买的时候相当于多花钱买了

        return value * (1 + commission_fee)

    # here we do not consider the level change when the positiion change is too big
    # we do consider the multi granity of our action and max holding case
    scale_factor = num_action - 1

    # init dp solution
    price_information = df.iloc[0]
    dp = [[0] * num_action for i in range(len(df))]
    for i in range(num_action):
        position_changed = (0 - i) / scale_factor * max_holding
        if position_changed > 0:
            # 要卖
            dp[0][i] = sell_value(price_information, position_changed)
        else:
            # 要买
            dp[0][i] = -buy_value(price_information, -position_changed)

    for i in range(1, len(df)):
        price_information = df.iloc[i]
        for j in range(num_action):
            # j是现在的选择
            previous_dp = []
            for k in range(num_action):
                # k是过去的选择
                position_changed = (k - j) / scale_factor * max_holding
                if position_changed > 0:
                    previous_dp.append(
                        dp[i - 1][k] +
                        sell_value(price_information, position_changed))
                else:
                    previous_dp.append(
                        dp[i - 1][k] -
                        buy_value(price_information, -position_changed))
            dp[i][j] = max(previous_dp)
    # 现在开始倒着取动作
    # 最后一个动作是清仓 看倒数第二个动作是怎么来的
    d1_dp_update = []
    for k in range(num_action):
        position_changed = (k - 0) / scale_factor * max_holding
        d1_dp_update.append(dp[len(df) - 2][k] +
                            sell_value(price_information, position_changed))
    last_action = d1_dp_update.index(dp[len(df) - 1][0])
    last_value = dp[len(df) - 2][last_action]
    action_list.append(last_action)
    for i in range(len(df) - 2, 0, -1):
        price_information = df.iloc[i]
        dn_dp_update = []
        for j in range(num_action):
            position_changed = (j - last_action) / scale_factor * max_holding
            if position_changed > 0:
                dn_dp_update.append(
                    dp[i - 1][j] +
                    sell_value(price_information, position_changed))
            else:
                dn_dp_update.append(
                    dp[i - 1][j] -
                    buy_value(price_information, -position_changed))
        current_action = dn_dp_update.index(last_value)
        last_action = current_action
        last_value = dp[i - 1][last_action]
        action_list.append(last_action)
    action_list.reverse()
    return action_list


def make_q_table(df: pd.DataFrame,
                 num_action,
                 max_holding,
                 commission_fee=0.000175,
                 max_punish=1e12):
    q_table = np.zeros((len(df) + 1, num_action, num_action))

    def sell_value(price_information, position):
        # use bid price and size to evaluate
        value = 0
        # position 表示剩余的单量
        for i in range(1, 6):
            if position < price_information["bid{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["bid{}_size".format(i)]
                value += price_information["bid{}_price".format(
                    i)] * price_information["bid{}_size".format(i)]
        if position > 0 and i == 5:
            value = value - max_punish
            # 执行的单量
        else:
            value += price_information["bid{}_price".format(i)] * position
        # 卖的时候的手续费相当于少卖钱了

        return value * (1 - commission_fee)

    def buy_value(price_information, position):
        # this value measure how much
        value = 0
        for i in range(1, 6):
            if position < price_information["ask{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["ask{}_size".format(i)]
                value += price_information["ask{}_price".format(
                    i)] * price_information["ask{}_size".format(i)]
        if i == 5 and position > 0:
            value = value + max_punish
        else:
            value += price_information["ask{}_price".format(i)] * position
        # 买的时候相当于多花钱买了

        return value * (1 + commission_fee)

    scale_factor = num_action - 1
    # calculate the Q_value in timestamp t-1 (starting from 0)
    price_information = df.iloc[-1]
    for i in range(num_action):
        position_change = (i - 0) / scale_factor * max_holding
        q_table[len(df) - 1][i][0] = sell_value(price_information,
                                                position_change)
    # 从后往前倒更新q值
    for t in range(2, len(df) + 1):
        price_information = df.iloc[-t]
        for current_position in range(num_action):
            for action in range(num_action):
                if action > current_position:
                    position_change = (action-current_position) / \
                        scale_factor*max_holding
                    q_table[len(df)-t][current_position][action] = - \
                        buy_value(price_information, position_change) + \
                        np.max(q_table[len(df)-t+1][action][:])
                else:
                    position_change = (current_position-action) / \
                        scale_factor*max_holding
                    q_table[len(df) -
                            t][current_position][action] = sell_value(
                                price_information, position_change) + np.max(
                                    q_table[len(df) - t + 1][action][:])
    return q_table


def make_q_table_reward(df: pd.DataFrame,
                        num_action,
                        max_holding,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.000175,
                        max_punish=1e12):
    q_table = np.zeros((len(df), num_action, num_action))

    # time * previous_action*current_action
    def sell_value(price_information, position):
        # use bid price and size to evaluate
        value = 0
        # position 表示剩余的单量
        for i in range(1, 6):
            if position <= price_information["bid{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["bid{}_size".format(i)]
                value += price_information["bid{}_price".format(
                    i)] * price_information["bid{}_size".format(i)]
        if position > 1e-12 and i == 5:
            value = value - max_punish
            # 执行的单量
        else:
            value += price_information["bid{}_price".format(i)] * position
        # 卖的时候的手续费相当于少卖钱了
        return value * (1 - commission_fee)

    def calculate_value(price_information, position):
        return price_information["bid1_price"] * position

    def buy_value(price_information, position):
        # this value measure how much
        value = 0
        for i in range(1, 6):
            if position <= price_information["ask{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["ask{}_size".format(i)]
                value += price_information["ask{}_price".format(
                    i)] * price_information["ask{}_size".format(i)]
        if i == 5 and position > 1e-12:
            value = value + max_punish
        else:
            value += price_information["ask{}_price".format(i)] * position
        # 买的时候相当于多花钱买了

        return value * (1 + commission_fee)

    scale_factor = num_action - 1
    # calculate the Q_value in timestamp t-1 (starting from 0)

    # 从后往前倒更新q值
    #TODO 更改reward的而不是单纯追求现金流
    for t in range(2, len(df) + 1):
        current_price_information = df.iloc[-t]
        future_price_information = df.iloc[-t + 1]
        for previous_action in range(num_action):
            for current_action in range(num_action):
                #the future_position is the current action
                if current_action > previous_action:
                    previous_position = previous_action / (
                        scale_factor) * max_holding
                    current_position = current_action / (
                        scale_factor) * max_holding
                    position_change = (current_action-previous_action) / \
                        scale_factor*max_holding
                    buy_money = buy_value(current_price_information,
                                          position_change)
                    current_value = calculate_value(current_price_information,
                                                    previous_position)
                    future_value = calculate_value(future_price_information,
                                                   current_position)
                    reward = future_value - (current_value + buy_money)
                    reward = reward_scale * reward
                    q_table[len(df) - t][previous_action][
                        current_action] = reward + gamma * np.max(
                            q_table[len(df) - t + 1][current_action][:])
                else:
                    previous_position = previous_action / (
                        scale_factor) * max_holding
                    current_position = current_action / (
                        scale_factor) * max_holding
                    position_change = (previous_action-current_action) / \
                        scale_factor*max_holding
                    sell_money = sell_value(current_price_information,
                                            position_change)
                    current_value = calculate_value(current_price_information,
                                                    previous_position)
                    future_value = calculate_value(future_price_information,
                                                   current_position)
                    reward = future_value + sell_money - current_value
                    reward = reward_scale * reward
                    q_table[len(df) - t][previous_action][
                        current_action] = reward + gamma * np.max(
                            q_table[len(df) - t + 1][current_action][:])
    return q_table


# get dp action from q table
def get_dp_action_from_qtable(q_table):
    action_list = []
    initial_q_table = q_table[0][0][:]
    first_action = np.argmax(initial_q_table)
    action_list.append(first_action)
    for i in range(1, len(q_table)):
        first_action = np.argmax(q_table[i, first_action, :])
        action_list.append(first_action)
    return action_list
