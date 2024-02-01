from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import os
import torch
import sys
#由于在for循环中 np计算会存有一点点剩余 导致剩余的position不全是0 进而导致出现买不全的现象
sys.path.append(".")

from tool.demonstration import making_multi_level_dp_demonstration, make_q_table, get_dp_action_from_qtable, make_q_table_reward
import math

tech_indicator_list = np.load("data/feature/second_feature.npy").tolist()
# tech_indicator_list=0
transcation_cost = 0.00
back_time_length = 1
max_holding_number = 0.01
action_dim = 5


class Testing_env(gym.Env):
    def __init__(self,
                 df: pd.DataFrame,
                 tech_indicator_list=tech_indicator_list,
                 transcation_cost=transcation_cost,
                 back_time_length=back_time_length,
                 max_holding_number=max_holding_number,
                 action_dim=action_dim,
                 early_stop=0,
                 initial_action=0):
        self.tech_indicator_list = tech_indicator_list
        self.df = df
        self.action_dim = action_dim
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list), ))
        self.terminal = False
        self.stack_length = back_time_length
        self.day = back_time_length
        self.data = self.df.iloc[self.day - self.stack_length:self.day]
        self.state = self.data[self.tech_indicator_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = 0
        self.position = 0
        self.position_holding_length = 1
        self.early_stop=early_stop
        self.initial_action=initial_action
        assert initial_action in range(action_dim)

    def sell_value(self, price_information, position):
        orgional_position = position
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
        if position > 1e-12 and i == 5:
            print("the holding could not be sell all clear")

            # 执行的单量
            actual_changed_position = orgional_position - position
        else:
            value += price_information["bid{}_price".format(i)] * position
            actual_changed_position = orgional_position
        # 卖的时候的手续费相当于少卖钱了
        self.comission_fee_history.append(self.comission_fee * value)

        return value * (1 - self.comission_fee), actual_changed_position

    def buy_value(self, price_information, position):
        # this value measure how much
        value = 0
        orgional_position = position
        for i in range(1, 6):
            if position < price_information["ask{}_size".format(i)] or i == 5:
                break
            else:
                position -= price_information["ask{}_size".format(i)]
                value += price_information["ask{}_price".format(
                    i)] * price_information["ask{}_size".format(i)]
        if i == 5 and position > 1e-12:
            print("the holding could not be bought all clear")
            actual_changed_position = orgional_position - position
            print("left order", position)
        else:
            value += price_information["ask{}_price".format(i)] * position
            actual_changed_position = orgional_position
        # 买的时候相当于多花钱买了

        self.comission_fee_history.append(self.comission_fee * value)

        return value * (1 + self.comission_fee), actual_changed_position

    def calculate_value(self, price_information, position):
        return price_information["bid1_price"] * position

    def calculate_avaliable_action(self, price_information):
        # 这块计算跟粒度有关系 修改粒度时应该注意
        buy_size_max = np.sum(price_information[[
            "ask1_size", "ask2_size", "ask3_size", "ask4_size"
        ]])
        sell_size_max = np.sum(price_information[[
            "bid1_size", "bid2_size", "bid3_size", "bid4_size"
        ]])
        position_upper = self.position + buy_size_max
        position_lower = self.position - sell_size_max
        position_lower = max(position_lower, 0)
        position_upper = min(position_upper, self.max_holding_number)
        # transfer the position back into our action
        current_action = int(self.position * (self.action_dim - 1) /
                             self.max_holding_number)
        action_upper = int(position_upper * (self.action_dim - 1) /
                           self.max_holding_number)
        if position_lower == 0:
            action_lower = 0
        else:
            #问题出在这里 如果可以被整除的话相当于少了一个动作
            action_lower = min(
                int(
                    math.ceil(position_lower * (self.action_dim - 1) /
                              self.max_holding_number) + 1e-12), action_upper,
                current_action)
        avaliable_discriminator = []
        for i in range(self.action_dim):
            if i >= action_lower and i <= action_upper:
                avaliable_discriminator.append(1)
            else:
                avaliable_discriminator.append(0)
        avaliable_discriminator = avaliable_discriminator
        return avaliable_discriminator

    def reset(self):
        self.terminal = False
        self.day = self.stack_length
        self.data = self.df.iloc[self.day - self.stack_length:self.day]
        self.state = self.data[self.tech_indicator_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        price_information = self.data.iloc[-1]
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        avaliable_discriminator = self.calculate_avaliable_action(
            price_information)
        self.previous_position = self.initial_action * self.max_holding_number / (
            self.action_dim - 1)
        self.position = self.initial_action * self.max_holding_number / (
            self.action_dim - 1)
        self.position_holding_length = 1
        self.needed_money_memory.append(self.position *
                                        self.data.iloc[-1]["ask1_price"])
        self.sell_money_memory.append(0)

        return self.state.reshape(-1), {
            "previous_action": self.initial_action,
            "avaliable_action": avaliable_discriminator,
            "holding_length": self.position_holding_length
        }

    def step(self, action):
        normlized_action = action / (self.action_dim - 1)
        position = self.max_holding_number * normlized_action
        # 目前没有future embedding day代表最新一天的信息
        self.terminal = (self.day >= len(self.df.index.unique()) - 1-self.early_stop)
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        self.day += 1
        self.data = self.df.iloc[self.day - self.stack_length:self.day]
        current_price_information = self.data.iloc[-1]
        self.state = self.data[self.tech_indicator_list].values
        self.previous_position = previous_position
        self.position = position
        if previous_position == position:
            self.position_holding_length += 1
        else:
            self.position_holding_length = 1

        if previous_position >= position:
            # hold the position or sell some position
            self.sell_size = previous_position - position

            cash, actual_position_change = self.sell_value(
                previous_price_information, self.sell_size)
            self.sell_money_memory.append(cash)
            self.needed_money_memory.append(0)
            self.position = self.previous_position - actual_position_change
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                 self.position)
            self.reward = current_value + cash - previous_value
            # 如果第一开始就是0而且没买
            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value + cash -
                               previous_value) / previous_value
            self.return_rate = return_rate
            self.reward_history.append(self.reward)

        if previous_position < position:
            # sell some of the position
            self.buy_size = position - previous_position
            needed_cash, actual_position_change = self.buy_value(
                previous_price_information, self.buy_size)
            self.needed_money_memory.append(needed_cash)
            self.sell_money_memory.append(0)

            self.position = self.previous_position + actual_position_change
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                 self.position)
            self.reward = current_value - needed_cash - previous_value
            return_rate = (current_value - needed_cash -
                           previous_value) / (previous_value + needed_cash)

            self.reward_history.append(self.reward)
            self.return_rate = return_rate
        self.previous_position = self.position
        avaliable_discriminator = self.calculate_avaliable_action(
            current_price_information)
        self.avaliable_discriminator = avaliable_discriminator
        # self.get_final_return_rate()
        # 检查是否出现return rate 为nan的情况
        if self.terminal:
            return_margin, pure_balance, required_money, commission_fee = self.get_final_return_rate(
            )
            self.pured_balance = pure_balance
            self.final_balance = self.pured_balance + self.calculate_value(
                current_price_information, self.position)
            self.required_money = required_money
            print("the portfit margine is ",
                  self.final_balance / (self.required_money + 1e-12))

        return self.state.reshape(-1), self.reward, self.terminal, {
            "previous_action": action,
            "avaliable_action": avaliable_discriminator,
            "holding_length": self.position_holding_length
        }

    def get_final_return_rate(self, slient=False):
        sell_money_memory = np.array(self.sell_money_memory)
        needed_money_memory = np.array(self.needed_money_memory)
        true_money = sell_money_memory - needed_money_memory
        final_balance = np.sum(true_money)
        balance_list = []
        for i in range(len(true_money)):
            balance_list.append(np.sum(true_money[:i + 1]))
        required_money = -np.min(balance_list)
        commission_fee = np.sum(self.comission_fee_history)
        return final_balance / (
            required_money +
            1e-12), final_balance, required_money, commission_fee


class Training_Env(Testing_env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        action_dim=action_dim,
        initial_action=0,
        gamma=1,
        reward_scale=10,
        early_end=3600,
    ):
        super(Training_Env,
              self).__init__(df, tech_indicator_list, transcation_cost,
                             back_time_length, max_holding_number, action_dim,early_end)
        self.q_table = make_q_table_reward(self.df,
                                           num_action=self.action_dim,
                                           max_holding=self.max_holding_number,
                                           commission_fee=transcation_cost,
                                           reward_scale=reward_scale,
                                           gamma=gamma,
                                           max_punish=1e12)
        self.reward_scale = reward_scale
        self.initial_action = initial_action
        self.position = self.initial_action * self.max_holding_number / (
            self.action_dim - 1)
        assert initial_action in range(action_dim)

    def reset(self):
        state, info = super(Training_Env, self).reset()
        info['previous_action_indicator'] = [0] * self.action_dim
        info['previous_action_indicator'][self.initial_action] = 1
        self.previous_action = self.initial_action
        self.previous_position = self.initial_action * self.max_holding_number / (
            self.action_dim - 1)
        self.position = self.initial_action * self.max_holding_number / (
            self.action_dim - 1)
        info['q_value'] = self.q_table[self.day - 1][self.previous_action][:]
        self.position = self.initial_action * self.max_holding_number / (
            self.action_dim - 1)
        info["previous_action"] = self.initial_action
        # self.needed_money_memory.append(self.position *
        #                                 self.data.iloc[-1]["ask1_price"])
        # self.sell_money_memory.append(0)
        return state, info

    def step(self, action):
        state, reward, done, info = super(Training_Env, self).step(action)

        info['q_value'] = self.q_table[self.day - 1][action][:]

        info['previous_action_indicator'] = [0] * self.action_dim
        info['previous_action_indicator'][action] = 1
        return state, reward * self.reward_scale, done, info


if __name__ == "__main__":
    data_path = "data/BTCUSDT/train/df_83.feather"
    df = pd.read_feather(data_path)
    env = Training_Env(df=df, initial_action=2, reward_scale=1)
    state, info = env.reset()
    reward_sum = 0
    done = False
    action_list = []
    while not done:
        action = np.argmax(info["q_value"])
        state, reward, done, info = env.step(action)
        reward_sum += reward
        action_list.append(action)
    print(reward_sum)
    np.save("env/initial_action_2/sell_money_memory.npy",env.sell_money_memory)
    np.save("env/initial_action_2/needed_money_memory.npy",env.needed_money_memory)
    np.save("env/initial_action_2/pure.npy",env.sell_money_memory)
    np.save("env/initial_action_2/pure_balance.npy",env.pured_balance)
    np.save("env/initial_action_2/final_balance.npy",env.final_balance)
    
    
    env = Training_Env(df=df, initial_action=1, reward_scale=1)
    state, info = env.reset()
    reward_sum = 0
    done = False
    action_list = []
    while not done:
        action = np.argmax(info["q_value"])
        state, reward, done, info = env.step(action)
        reward_sum += reward
        action_list.append(action)
    print(reward_sum)
    np.save("env/initial_action_1/sell_money_memory.npy",env.sell_money_memory)
    np.save("env/initial_action_1/needed_money_memory.npy",env.needed_money_memory)
    np.save("env/initial_action_1/pure_balance.npy",env.pured_balance)
    np.save("env/initial_action_1/final_balance.npy",env.final_balance)
