import pandas as pd
import numpy as np
import warnings
import sys

sys.path.append(".")
from tool import market_dynamics_modeling_analysis
from tool import label_util as util
from comprehensive_evaluation.analyzer import *


class Analyzer:
    def __init__(self, market_information, strategy):
        # market information should be a data frame that consist of 21 columns(timedstamp &  5-level ask bid's price and size)
        # stategy: a list of dictionary consisting of 3 keys: timestamp, action (buy or sell) and dictionary indicating the level
        # price and amount of the conducted orders
        self.market_information = market_information
        self.strategy = strategy
        # order in strategy[{'timestamp':,'action':,'order':[{'price':,'amount':},{'price':,'amount':},...],'position':}]
        # the strategy should be a list of market order containing the executed price, amount the agent's position after conducting the trade
        # check the pricing the problem
        price_timing = [price["timestamp"] for price in self.strategy]
        assert max(price_timing) <= max(self.market_information.timestamp.unqiue())
        # check the price is legal
        for timestamp in self.market_information.timestamp.unique():
            price_single_timestamp = self.market_information[
                self.market_information["timestamp"] == timestamp
            ]
            assert (
                price_single_timestamp["ask1_price"]
                >= price_single_timestamp["bid1_price"]
            )
        # check the strategy opening position is whether is over rated and place correctly
        # 对于买单我们要检查他的买入价格的下限应该符合ask1 price
        for stack_order in strategy:
            timestamp = stack_order["timestamp"]
            current_market_information = self.market_information[
                self.market_information["timestamp"] == timestamp
            ]
            assert stack_order["action"] in ["buy", "sell"]
            if stack_order["action"] == "buy":
                list_order = stack_order["order"]
                level_number = len(list_order)
                for i in range(level_number):
                    assert (
                        list_order[i]["price"]
                        == current_market_information["ask{}_price".format(i)].values[0]
                    )
                    assert (
                        list_order[i]["size"]
                        <= current_market_information["ask{}_size".format(i)].values[0]
                    )
            elif stack_order["action"] == "sell":
                list_order = stack_order["order"]
                level_number = len(list_order)
                for i in range(level_number):
                    assert (
                        list_order[i]["price"]
                        == current_market_information["bid{}_price".format(i)].values[0]
                    )
                    assert (
                        list_order[i]["size"]
                        <= current_market_information["bid{}_size".format(i)].values[0]
                    )
        # check the trace of the position in the trading process is legal or not. it always should be 0 at the start and end of the trading process
        if self.strategy[-1]["position"] != 0:
            last_position = self.strategy[-1]["position"]
            warnings.warn(
                "the final position of the strategy is not zero, we force the agent to close its position in the last timestamp"
            )
            last_market_information = self.market_information[
                self.market_information["timestamp"]
                == max(self.market_information["timestamp"].unique()[-1])
            ]
            size_sum = 0
            if (
                last_position
                > last_market_information["bid1_size"].values[0]
                + last_market_information["bid2_size"].values[0]
                + last_market_information["bid3_size"].values[0]
                + last_market_information["bid4_size"].values[0]
                + last_market_information["bid5_size"].values[0]
            ):
                warnings.warn(
                    "we barely trade at this timstamp instantly because there is no enough liquidity in the market,\
                we force the agent to close its position in the last timestamp by expanding the last level's size"
                )
                last_market_information["bid5_size"] = last_position - (
                    last_market_information["bid1_size"].values[0]
                    + last_market_information["bid2_size"].values[0]
                    + last_market_information["bid3_size"].values[0]
                    + last_market_information["bid4_size"].values[0]
                )
            for i in range(5):
                size_sum += last_market_information["bid{}_size".format(i + 1)].values[
                    0
                ]
                if last_position <= size_sum:
                    break
            level_order_size_list = []
            order_remaining = last_position
            for j in range(i + 1):
                level_order_size_list.append(
                    min(
                        order_remaining,
                        last_market_information["bid{}_size".format(j + 1)].values[0],
                    )
                )
                order_remaining.append(
                    order_remaining
                    - last_market_information["bid{}_size".format(j + 1)].values[0]
                )

    def analysis_behavior(self, selected_strategy):
        # 现确定总共的开闭仓的次数 selected strategy 起码开头和结尾的position应该为0
        opening_strategy_timestamp_list = []
        closing_strategy_timestamp_list = []

        for stack_order in selected_strategy:
            if stack_order["action"] == "buy":
                order_size = 0
                for order in stack_order["order"]:
                    order_size += order["size"]
                if order_size == stack_order["position"]:
                    opening_strategy_timestamp_list.append(stack_order["timestamp"])
            elif stack_order["action"] == "sell":
                if stack_order["position"] == 0:
                    closing_strategy_timestamp_list.append(stack_order["timestamp"])
        assert len(opening_strategy_timestamp_list) == len(
            closing_strategy_timestamp_list
        )  # 确保开仓和平仓的次数相同
        trade_timestamp_list = list(
            zip(opening_strategy_timestamp_list, closing_strategy_timestamp_list)
        )

        # 1. 计算每次交易的收益率以及开仓到平仓的时间
        total_return_rate = 0
        total_duration = 0
        total_mdd = 0

        for open_time, close_time in trade_timestamp_list:
            assert open_time < close_time
            single_trade_strategy = []
            for selected_stack_order in selected_strategy:
                if (
                    selected_stack_order["timestamp"] >= open_time
                    and selected_stack_order["timestamp"] <= close_time
                ):
                    single_trade_strategy.append(selected_stack_order)
            cash_flow = []
            # 计算每次交易的现金流的变化
            for stack_order in single_trade_strategy:
                total_value = 0
                for order in stack_order["order"]:
                    total_value += order["price"] * order["amount"]
                    if stack_order["action"] == "buy":
                        total_value = -total_value
                    cash_flow.append(total_value)
            cash_record = [sum(cash_flow[: i + 1]) for i in range(len(cash_flow))]
            final_cash = cash_record[-1]
            require_money = -min(cash_record)
            # 计算每次交易的收益率和持仓时间
            return_rate = final_cash / require_money
            total_return_rate += return_rate
            total_duration += close_time - open_time
            # TODO 根据bid1 price进行结算，每次持仓过程中的maxdrawdown
            position_record = []
            timestamp_record = []
            trade_position_record = []
            cash_accmulative_record = []
            for stack_order in single_trade_strategy:
                timestamp_record.append(stack_order["timestamp"])
                position_record.append(stack_order["position"])
            corresponding_market_timestamp = [
                timestamp
                for timestamp in self.market_information["timestamp"].unique()
                if timestamp >= open_time and timestamp <= close_time
            ]
            assert len(timestamp_record) == len(position_record)
            assert len(timestamp_record) == len(cash_record)
            for i in range(len(timestamp_record) - 1):
                time_point = [
                    timestamp
                    for timestamp in self.market_information["timestamp"].unique()
                    if timestamp >= timestamp_record[i]
                    and timestamp <= timestamp_record[i + 1]
                ]
                cash_accmulative_record.append(cash_record[i] + require_money)
                for j in range(len(time_point)):
                    trade_position_record.append(position_record[i])
                for k in range(len(time_point) - 1):
                    cash_accmulative_record.append(cash_accmulative_record[-1])
            trade_position_record.append(0)
            corresponding_market_information = self.market_information[
                self.market_information["timestamp"].isin(
                    corresponding_market_timestamp
                )
            ]
            assert len(trade_position_record) == len(corresponding_market_information)
            position_value_record = [
                position * single_value
                for position, single_value in zip(
                    trade_position_record, corresponding_market_information
                )
            ]
            total_value_record = [
                cash + position_value
                for cash, position_value in zip(
                    cash_accmulative_record, position_value_record
                )
            ]
            mdd = 0
            peak = total_value_record[0]
            for value in total_value_record:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > mdd:
                    mdd = dd
            total_mdd += mdd

        mean_return_rate = total_return_rate / len(trade_timestamp_list)
        mean_duration = total_duration / len(trade_timestamp_list)
        mean_mdd = total_mdd / len(trade_timestamp_list)
        return mean_return_rate, mean_duration, mean_mdd

    def calculate_metric(self, selected_timestamp: list):
        # selected trade is part of the strategy that we want to calculate the metric,
        # its position but do not have to end or start with 0
        # selected_timestamp is a 2-element list indicating the start and end of the timestamp
        selected_timestamp.sort()
        assert len(selected_timestamp) == 2
        selected_market = self.market_information[
            (self.market_information["timestamp"] >= selected_timestamp[0])
            & (self.market_information["timestamp"] <= selected_timestamp[1])
        ]
        selected_strategy = [
            item
            for item in self.strategy
            if selected_timestamp[0] <= item["timestamp"] <= selected_timestamp[1]
        ]
        first_trade = selected_strategy[0]
        first_trade_size = sum([level["amount"] for level in first_trade["order"]])
        if first_trade["action"] == "buy":
            first_trade_size = -first_trade_size

        initial_posotion = selected_strategy[0]["position"] + first_trade_size
        assert initial_posotion >= 0
        cash_flow = []
        for stack_order in selected_strategy:
            total_value = 0
            for order in stack_order["order"]:
                total_value += order["price"] * order["amount"]
                if stack_order["action"] == "buy":
                    total_value = -total_value
                cash_flow.append(total_value)
        initial_require_money = (
            initial_posotion * selected_market["bid1_price"].values[0]
        )
        cash_record = [sum(cash_flow[: i + 1]) for i in range(len(cash_flow))]
        require_money = initial_require_money - min(0, min(cash_record))
        position = initial_posotion
        position_market_record = []
        cash_market_record = []
        for timestamp in selected_market.timestamp.unique():
            matching_strategy = next(
                (item for item in selected_strategy if item["timestamp"] == timestamp),
                None,
            )
            if matching_strategy:
                current_position = matching_strategy["position"]
                position = current_position
            else:
                current_position = position
            position_market_record.append(current_position)
        inital_cash = require_money - initial_require_money
        cash = inital_cash
        for timestamp in selected_market.timestamp.unique():
            matching_strategy = next(
                (item for item in selected_strategy if item["timestamp"] == timestamp),
                None,
            )
            if matching_strategy:
                total_value = 0
                for order in matching_strategy["order"]:
                    total_value += order["price"] * order["amount"]
                    if stack_order["action"] == "buy":
                        total_value = -total_value
                current_cash = cash + total_value
                cash = current_cash
            else:
                current_cash = cash
            cash_market_record.append(current_cash)
        assert len(position_market_record) == len(selected_market)
        selected_market_price = selected_market["bid1_price"].values
        position_value_record = [
            position * single_value
            for position, single_value in zip(
                position_market_record, selected_market_price
            )
        ]
        total_value_record = [
            cash + position_value
            for cash, position_value in zip(cash_market_record, position_value_record)
        ]
        tr = total_value_record[-1] / total_value_record[0] - 1
        mdd = 0
        peak = total_value_record[0]
        for value in total_value_record:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > mdd:
                mdd = dd
        cr = tr / mdd
        return tr, mdd, cr

    def analysis_along_time(self, num_seg):
        opening_strategy_timestamp_list = []
        closing_strategy_timestamp_list = []

        for stack_order in self.strategy:
            if stack_order["action"] == "buy":
                order_size = 0
                for order in stack_order["order"]:
                    order_size += order["size"]
                if order_size == stack_order["position"]:
                    opening_strategy_timestamp_list.append(stack_order["timestamp"])
            elif stack_order["action"] == "sell":
                if stack_order["position"] == 0:
                    closing_strategy_timestamp_list.append(stack_order["timestamp"])
        assert len(opening_strategy_timestamp_list) == len(
            closing_strategy_timestamp_list
        )

        assert len(opening_strategy_timestamp_list) >= num_seg
        num_trade_seg = int(len(opening_strategy_timestamp_list) / num_seg)
        for i in range(num_seg):
            if i != num_seg - 1:
                selected_timestamp = [
                    opening_strategy_timestamp_list[num_trade_seg * i],
                    closing_strategy_timestamp_list[num_trade_seg * (i + 1)],
                ]
                tr, mdd, cr = self.calculate_metric(selected_timestamp)

                print(
                    "in the {}th segment, the total return rate is {}, the max drawdown is {}, the calmar ratio is {}".format(
                        i, tr, mdd, cr
                    )
                )

            else:
                selected_timestamp = [
                    opening_strategy_timestamp_list[num_trade_seg * i],
                    closing_strategy_timestamp_list[-1],
                ]

                tr, mdd, cr = self.calculate_metric(selected_timestamp)
                print(
                    "in the {}th segment, the total return rate is {}, the max drawdown is {}, the calmar ratio is {}".format(
                        i, tr, mdd, cr
                    )
                )

    def analysis_along_dynamics(self, num_dynamic):
        # 解决开平仓的不在一个dynamics的问题
        # 统计在一个时段内的动作买入 买入量 在不同的dynamics上 不看具体的trading behavior 关注交易总量中 买入/卖出量在某个特定dynamics上的占比

        pass


if __name__ == "__main__":
    print("hello world")
