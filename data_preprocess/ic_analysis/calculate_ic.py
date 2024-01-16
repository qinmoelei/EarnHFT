import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default=
    "preprocess/merge/BTCUSDT/2023-10-08-2023-10-18/df.feather",
    help="the path of storing the data")

parser.add_argument("--save_path",
                    type=str,
                    default="ic_analysis/feature_analysis",
                    help="the path of storing the data")
reward_features = [
    "timestamp",
    "symbol",
    "bid1_size",
    "ask1_size",
    "bid1_price",
    "ask1_price",
    "bid2_size",
    "ask2_size",
    "bid2_price",
    "ask2_price",
    "bid3_size",
    "ask3_size",
    "bid3_price",
    "ask3_price",
    "bid4_size",
    "ask4_size",
    "bid4_price",
    "ask4_price",
    "bid5_size",
    "ask5_size",
    "bid5_price",
    "ask5_price",
]


def find_dates_tic(data_path):
    pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    matches = pattern.findall(data_path)
    start_date = matches[0]
    end_date = matches[1]
    ticker = data_path.split("/")[-3]

    return start_date, end_date, ticker


def analysis_ic_longterm(df: pd.DataFrame, period=1, theshold=0.01):
    reward_features = [
        "timestamp",
        "symbol",
        "bid1_size",
        "ask1_size",
        "bid1_price",
        "ask1_price",
        "bid2_size",
        "ask2_size",
        "bid2_price",
        "ask2_price",
        "bid3_size",
        "ask3_size",
        "bid3_price",
        "ask3_price",
        "bid4_size",
        "ask4_size",
        "bid4_price",
        "ask4_price",
        "bid5_size",
        "ask5_size",
        "bid5_price",
        "ask5_price",
        "open",
        "high",
        "low",
        "close",
        "midpoint",
    ]
    feature = df.columns.tolist()
    feature = list(set(feature).difference(set(reward_features)))
    return_rate = df["bid1_price"].diff(periods=period).tolist()
    for i in range(period):
        return_rate.pop(0)
        return_rate.append(0)
    df["return"] = return_rate
    cor = dict()
    for f in tqdm(feature):
        correlation = df["return"].corr(df[f])
        cor["{}".format(f)] = correlation
    for key in cor:
        cor[key] = np.abs(cor[key])
    feature_train = sorted(cor.items(), key=lambda x: x[1], reverse=True)
    feature_train = dict(feature_train)
    presever_features = []
    for key in feature_train:
        if feature_train[key] >= theshold:
            presever_features.append(key)
    return feature_train, presever_features


def get_analysis_result(args):
    df = pd.read_feather(args.data_path)
    start_date, end_date, ticker = find_dates_tic(args.data_path)
    feature_train, presever_features = analysis_ic_longterm(df ,period=1, theshold=0.01)

    if not os.path.exists(
            os.path.join(args.save_path, ticker, "{}_{}".format(
                start_date, end_date))):
        os.makedirs(
            os.path.join(args.save_path, ticker,
                         "{}_{}".format(start_date, end_date)))
    path = os.path.join(args.save_path, ticker,
                        "{}_{}".format(start_date, end_date))
    np.save(
        os.path.join(
            path,
            "second.npy"),
        feature_train)
    np.save(
        os.path.join(
            path,
            "second_feature.npy"),
        presever_features)
    
    feature_train, presever_features = analysis_ic_longterm(df ,period=60, theshold=0.01)

    np.save(
        os.path.join(
            path,
            "minitue.npy"),
        feature_train)
    np.save(
        os.path.join(
            path,
            "minitue_feature.npy"),
        presever_features)


if __name__ == "__main__":
    args = parser.parse_args()
    get_analysis_result(args)