import pandas as pd
import numpy as np
import argparse
import re
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="preprocess/create_features_new/BTCTUSD/2023-03-30-2023-05-10",
    help="the path of storing the data")

parser.add_argument("--save_path",
                    type=str,
                    default="preprocess/merge",
                    help="the path of storing the data")


def find_dates_tic(data_path):
    pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    matches = pattern.findall(data_path)
    start_date = matches[0]
    end_date = matches[1]
    ticker = data_path.split("/")[-3]

    return start_date, end_date, ticker


def merge(args):
    order_book_path = os.path.join(args.data_path, "orderbook.feather")
    trade_minitue_path = os.path.join(args.data_path, "trade_minitue.feather")
    trade_second_path = os.path.join(args.data_path, "trade_second.feather")
    order_book = pd.read_feather(order_book_path)
    trade_minitue = pd.read_feather(trade_minitue_path)
    trade_second = pd.read_feather(trade_second_path)
    order_book = order_book.set_index("timestamp")
    trade_minitue = trade_minitue.set_index("timestamp")
    trade_second = trade_second.set_index("timestamp")
    #merge second & minitue trade
    trade_minitue = trade_minitue.resample('S').ffill()
    trade = pd.concat([trade_second, trade_minitue], axis=1)
    start_time = max(trade_second.index[0], trade_minitue.index[0])
    end_time = min(trade_second.index[-1], trade_minitue.index[-1])
    trade = trade.loc[start_time:end_time]
    #merge trade & oe
    df = pd.merge(order_book, trade, left_index=True, right_index=True)
    #check for nan
    df = df.resample('1S').asfreq()
    df = df.ffill().reset_index()
    start_date, end_date, ticker = find_dates_tic(args.data_path)
    if not os.path.exists(
            os.path.join(args.save_path, ticker, "{}-{}".format(
                start_date, end_date))):
        os.makedirs(
            os.path.join(args.save_path, ticker,
                         "{}-{}".format(start_date, end_date)))
    df.to_feather(os.path.join(args.save_path, ticker,
                         "{}-{}".format(start_date, end_date),"df.feather"))
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    merge(args)
