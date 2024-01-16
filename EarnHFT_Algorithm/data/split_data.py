import pandas as pd
import numpy as np
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="data/BTCTUSD",
    help="the path for the orgional df",
)

parser.add_argument(
    "--chunk_length",
    type=int,
    default=14400,
    help="the chunk size of each df",
)

parser.add_argument(
    "--future_sight",
    type=int,
    default=3600,
    help="the chunk size of each df",
)


def split_data(df: pd.DataFrame):
    train_length = int(len(df) * 0.6)
    valid_length = int(len(df) * 0.2)
    test_length = int(len(df) * 0.2)
    train = df.iloc[:train_length].reset_index(drop=True)
    valid = df.iloc[train_length : train_length + valid_length].reset_index(drop=True)
    test = df.iloc[
        train_length + valid_length : train_length + valid_length + test_length
    ].reset_index(drop=True)
    return train, valid, test


def save_split_data(args):
    df = pd.read_feather(os.path.join(args.data_path, "df.feather"))
    train, valid, test = split_data(df)
    train.to_feather(os.path.join(args.data_path, "train.feather"))
    valid.to_feather(os.path.join(args.data_path, "valid.feather"))
    test.to_feather(os.path.join(args.data_path, "test.feather"))
    train_list = [
        train.iloc[i : i + args.chunk_length + args.future_sight].reset_index(drop=True)
        for i in range(0, len(train), args.chunk_length)
    ]
    if not os.path.exists(os.path.join(args.data_path,"train")):
        os.makedirs(os.path.join(args.data_path,"train"))
    for i in range(len(train_list)):
        train_list[i].to_feather(os.path.join(args.data_path,"train","df_{}.feather".format(i)))

if __name__=="__main__":
    args = parser.parse_args()
    save_split_data(args)