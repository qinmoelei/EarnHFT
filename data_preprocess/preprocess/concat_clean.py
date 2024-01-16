import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    type=str,
                    default="download_from_tardis",
                    help="the path of storing the data")

parser.add_argument("--symbols",
                    type=str,
                    default="BTCTUSD",
                    help="the name of the ticker")

parser.add_argument("--start_date",
                    type=str,
                    default="2023-03-30",
                    help="the date of start")

parser.add_argument("--end_date",
                    type=str,
                    default="2023-04-05",
                    help="the date of end")
parser.add_argument("--save_path",
                    type=str,
                    default="preprocess/concat_clean",
                    help="the path to save the data")

############################################### new preprocess#########################################################
def find_strings_in_range(string_list, start_time, end_time):
    result = []

    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    for string in string_list:
        matches = pattern.findall(string)
        for match in matches:
            try:
                time = datetime.strptime(match, '%Y-%m-%d')
                if start_time <= time <= end_time:
                    result.append(string)
            except ValueError:
                pass
    return result


#clean order book
def get_order_book_data(args):
    orderbook_dir = "{}/{}/book_snapshot_5".format(args.data_path,
                                                   args.symbols)
    order_book_file = os.listdir(orderbook_dir)
    order_book_file.sort()
    filter_order_book_file = find_strings_in_range(order_book_file,
                                                   args.start_date,
                                                   args.end_date)
    df_list = []
    for file in filter_order_book_file:
        df_list.append(
            pd.read_csv(os.path.join(orderbook_dir, file), engine='python'))
    orderbook_df = pd.concat(df_list, axis=0)
    orderbook_df = orderbook_df.drop(columns=["exchange", "local_timestamp"])
    orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'] *
                                               1000)
    orderbook_df = orderbook_df.set_index('timestamp')
    orderbook_df = orderbook_df.groupby([pd.Grouper(freq='S')]).agg(['first'])
    for i in range(5):
        orderbook_df["ask{}_price".format(i + 1)] = orderbook_df.pop(
            "asks[{}].price".format(i))
        orderbook_df["ask{}_size".format(i + 1)] = orderbook_df.pop(
            "asks[{}].amount".format(i))
        orderbook_df["bid{}_price".format(i + 1)] = orderbook_df.pop(
            "bids[{}].price".format(i))
        orderbook_df["bid{}_size".format(i + 1)] = orderbook_df.pop(
            "bids[{}].amount".format(i))
    orderbook_df.columns = orderbook_df.columns.droplevel(level=1)
    orderbook_df = orderbook_df.reset_index()
    orderbook_df = scratch_order_book_information(orderbook_df)
    return orderbook_df


def scratch_order_book_information(df: pd.DataFrame):
    df = df.set_index('timestamp')

    # Resample the data at one-second frequency
    df = df.resample('1S').asfreq()

    # Forward-fill the missing values
    df = df.ffill().reset_index()
    return df


def scratch_trade_information(df: pd.DataFrame,time_diff):
    df = df.set_index('timestamp')

    # Resample the data at one-second frequency
    df = df.resample(time_diff).asfreq()

    # Forward-fill the missing values
    df = df.ffill().reset_index()
    return df



def get_trade_data(args):
    trades_dir = "{}/{}/trades".format(args.data_path, args.symbols)
    trades_file = os.listdir(trades_dir)
    trades_file.sort()
    filter_trades_file = find_strings_in_range(trades_file, args.start_date,
                                               args.end_date)
    df_list = []
    for file in filter_trades_file:
        df_list.append(pd.read_csv(os.path.join(trades_dir, file)))
    trades_df = pd.concat(df_list, axis=0)
    trades_df = trades_df.drop(columns=["exchange", "local_timestamp"])
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'] * 1000)
    trades_df = trades_df.set_index('timestamp')
    trades_df_second = trades_df.groupby([pd.Grouper(freq='S')])
    trades_df_minitue = trades_df.groupby([pd.Grouper(freq='1Min')])
    trades_df_second = compress_trade_information(
        trades_df_second, cols=["open_s", "high_s", "low_s", "close_s"])
    trades_df_minitue = compress_trade_information(
        trades_df_minitue, cols=["open_m", "high_m", "low_m", "close_m"])
    trades_df_second = trades_df_second.set_index('timestamp')
    trades_df_minitue = trades_df_minitue.set_index('timestamp')
    trades_df_minitue.index = trades_df_minitue.index + pd.Timedelta(minutes=1)
    trades_df_second.index = trades_df_second.index + pd.Timedelta(seconds=1)
    #timestamp represents the timestamp that the trade could see
    trades_df_second = trades_df_second.reset_index()
    trades_df_minitue = trades_df_minitue.reset_index()
    
    trades_df_second=scratch_trade_information(trades_df_second,'1S')
    trades_df_minitue=scratch_trade_information(trades_df_minitue,'1T')
    return trades_df_minitue, trades_df_second


def compress_trade_information(df_group_by,
                               cols=[
                                   "open",
                                   "high",
                                   "low",
                                   "close",
                               ]):
    agg = df_group_by.agg({
        "price": ["first", "max", "min", "last"],
    })
    output = pd.DataFrame(agg.values, columns=cols)
    output["timestamp"] = agg.index.values
    output = output[["timestamp"] + cols]
    return output


def get_downloaded_data(args):
    orderbook_df = get_order_book_data(args)

    trades_df_minitue, trades_df_second = get_trade_data(args)
    if not os.path.exists(
            os.path.join(args.save_path, args.symbols, "{}-{}".format(
                args.start_date, args.end_date))):
        os.makedirs(
            os.path.join(args.save_path, args.symbols,
                         "{}-{}".format(args.start_date, args.end_date)))
    orderbook_df.to_feather(
        os.path.join(args.save_path, args.symbols,
                     "{}-{}".format(args.start_date,
                                    args.end_date), "orderbook.feather"))
    trades_df_minitue.to_feather(
        os.path.join(args.save_path, args.symbols,
                     "{}-{}".format(args.start_date,
                                    args.end_date), "trade_minitue.feather"))
    trades_df_second.to_feather(
        os.path.join(args.save_path, args.symbols,
                     "{}-{}".format(args.start_date,
                                    args.end_date), "trade_second.feather"))
    return orderbook_df, trades_df_minitue, trades_df_second

############################################### new preprocess#########################################################


if __name__ == "__main__":
    args = parser.parse_args()
    get_downloaded_data(args)