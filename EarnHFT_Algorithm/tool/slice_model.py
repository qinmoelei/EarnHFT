import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
import pandas as pd
import argparse
from pathlib import Path
import warnings
import market_dynamics_modeling_analysis
import label_util as util

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--data_path",
    type=str,
    default="data/BTCTUSD/valid.feather",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--key_indicator",
    type=str,
    default="bid1_price",
    help="The column name of the feature in the data that will be used for dynamic modeling",
)
parser.add_argument(
    "--timestamp",
    type=str,
    default="index",
    help="The column name of the feature in the data that is the timestamp",
)
parser.add_argument(
    "--tic",
    type=str,
    default="tic",
    help="The column name of the feature in the data that marks the tic",
)
parser.add_argument(
    "--labeling_method",
    type=str,
    default="slope",
    help="The method that is used for dynamic labeling:quantile/slope/DTW",
)
parser.add_argument(
    "--min_length_limit",
    type=int,
    default=60,
    help="Every slice will have at least this length",
)
parser.add_argument(
    "--merging_metric",
    type=str,
    default="DTW_distance",
    help="The method that is used for slice merging",
)
parser.add_argument(
    "--merging_threshold",
    type=float,
    default=0.0003,
    help="The metric threshold that is used to decide whether a slice will be merged",
)
parser.add_argument(
    "--merging_dynamic_constraint",
    type=int,
    default=1,
    help="Neighbor segment of dynamics spans greater than this number will not be merged(setting this to $-1$ will disable the constraint)",
)
parser.add_argument(
    "--filter_strength",
    type=int,
    default=1,
    help='The strength of the low-pass Butterworth filter, the bigger the lower cutoff frequency, "1" have the cutoff frequency of min_length_limit period',
)
parser.add_argument(
    "--dynamic_number",
    type=int,
    default=5,
    help='The strength of the low-pass Butterworth filter, the bigger the lower cutoff frequency, "1" have the cutoff frequency of min_length_limit period',
)
parser.add_argument(
    "--max_length_expectation",
    type=int,
    default=3600,
    help='The strength of the low-pass Butterworth filter, the bigger the lower cutoff frequency, "1" have the cutoff frequency of min_length_limit period',
)


class Linear_Market_Dynamics_Model(object):
    def __init__(self, args):
        super(Linear_Market_Dynamics_Model, self).__init__()
        self.data_path = args.data_path
        self.method = "slice_and_merge"
        self.filter_strength = args.filter_strength
        self.dynamic_number = args.dynamic_number
        self.max_length_expectation = args.max_length_expectation
        self.key_indicator = args.key_indicator
        self.timestamp = args.timestamp
        self.tic = args.tic
        self.labeling_method = args.labeling_method
        self.min_length_limit = args.min_length_limit
        self.merging_metric = args.merging_metric
        self.merging_threshold = args.merging_threshold
        self.merging_dynamic_constraint = args.merging_dynamic_constraint

    def file_extension_selector(self, read):
        if self.data_path.endswith(".csv"):
            if read:
                return pd.read_csv
            else:
                return pd.DataFrame.to_feather
        elif self.data_path.endswith(".feather"):
            if read:
                return pd.read_feather
            else:
                return pd.DataFrame.to_feather
        else:
            raise ValueError("invalid file extension")

    def wirte_data_as_segments(self, data, process_datafile_path):
        # get file name and extension from process_datafile_path
        file_name, file_extension = os.path.splitext(process_datafile_path)

    def run(self):
        print("labeling start")
        path_names = Path(self.data_path).resolve().parents
        ticker_name_path = path_names[0]
        output_path = self.data_path
        raw_data = pd.read_feather(self.data_path)
        raw_data[self.tic] = raw_data["symbol"]
        raw_data[self.key_indicator] = raw_data["bid1_price"]
        raw_data[self.timestamp] = raw_data.index
        process_data_path = os.path.join(ticker_name_path, "valid_processed.feather")
        raw_data.to_feather(process_data_path)
        self.data_path = process_data_path

        worker = util.Worker(
            self.data_path,
            "slice_and_merge",
            filter_strength=self.filter_strength,
            key_indicator=self.key_indicator,
            timestamp=self.timestamp,
            tic=self.tic,
            labeling_method=self.labeling_method,
            min_length_limit=self.min_length_limit,
            merging_threshold=self.merging_threshold,
            merging_metric=self.merging_metric,
            merging_dynamic_constraint=self.merging_dynamic_constraint,
        )
        print("start fitting")
        worker.fit(
            self.dynamic_number, self.max_length_expectation, self.min_length_limit
        )
        print("finish fitting")
        worker.label(os.path.dirname(self.data_path), final=True)
        labeled_data = pd.concat([v for v in worker.data_dict.values()], axis=0)
        flie_reader = self.file_extension_selector(read=True)
        extension = self.data_path.split(".")[-1]
        data = flie_reader(self.data_path)
        if self.tic in data.columns:
            merge_keys = [self.timestamp, self.tic, self.key_indicator]
        else:
            merge_keys = [self.timestamp, self.key_indicator]
        merged_data = data.merge(
            labeled_data, how="left", on=merge_keys, suffixes=("", "_DROP")
        ).filter(regex="^(?!.*_DROP)")
        if self.labeling_method == "slope":
            self.model_id = f"slice_and_merge_model_{self.dynamic_number}dynamics_minlength{self.min_length_limit}_{self.labeling_method}_labeling_slope"
        else:
            self.model_id = f"slice_and_merge_model_{self.dynamic_number}dynamics_minlength{self.min_length_limit}_{self.labeling_method}_labeling"

        process_datafile_path = (
            os.path.splitext(output_path)[0]
            + "_labeled_"
            + self.model_id
            + "."
            + extension
        )
        # if extension == "csv":
        #     merged_data.to_csv(process_datafile_path, index=False)
        # elif extension == "feather":
        #     merged_data.to_feather(process_datafile_path)
        print("labeling done")
        if not os.path.exists(os.path.join(ticker_name_path, "valid")):
            os.makedirs(os.path.join(ticker_name_path, "valid"))
        for i in range(self.dynamic_number):
            os.makedirs(os.path.join(ticker_name_path, "valid", "label_{}".format(i)))
        previous_label = merged_data.label[0]
        previous_start = 0
        label_counter = [0] * self.dynamic_number
        for i in range(len(merged_data)):
            if merged_data.label[i] != previous_label:
                merged_data.iloc[previous_start:i].reset_index(drop=True).to_feather(
                    os.path.join(
                        ticker_name_path,
                        "valid",
                        "label_{}".format(previous_label),
                        "df_{}.feather".format(label_counter[previous_label]),
                    )
                )
                label_counter[previous_label] += 1
                previous_start = i
                previous_label = merged_data.label[i]

        # print("plotting start")
        # # a list the path to all the modeling visulizations
        # market_dynamic_labeling_visualization_paths = worker.plot(
        #     worker.tics, self.slope_interval, output_path, self.model_id
        # )
        # print("plotting done")
        # # if self.OE_BTC == True:
        # #     os.remove('./temp/OE_BTC_processed.csv')

        # # MDM analysis
        # MDM_analysis = market_dynamics_modeling_analysis.MarketDynamicsModelingAnalysis(
        #     process_datafile_path, self.key_indicator
        # )
        # MDM_analysis.run_analysis(process_datafile_path)
        # print("Market dynamics modeling analysis done")

        # return (
        #     os.path.abspath(process_datafile_path),
        #     market_dynamic_labeling_visualization_paths,
        # )


if __name__ == "__main__":
    args = parser.parse_args()
    model = Linear_Market_Dynamics_Model(args)
    model.run()
