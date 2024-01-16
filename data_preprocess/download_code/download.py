import nest_asyncio
from tardis_dev import datasets, get_exchange_details
import logging
import os, tarfile
import gzip
import shutil

nest_asyncio.apply()
logging.basicConfig(level=logging.DEBUG)

# function used by default if not provided via options
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--symbols",
    type=str,
    default="BTCUSDT",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--start_date", type=str, default="2023-04-01", help="start_date of the downloading"
)
parser.add_argument(
    "--end_date", type=str, default="2023-10-18", help="end_date of the downloading"
)

parser.add_argument(
    "--data_types",
    type=str,
    default="book_snapshot_5",
    help="the number of transcation we store in one memory",
)
args = parser.parse_args()


def default_file_name(exchange, data_type, date, symbol, format):
    return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# customized get filename function - saves data in nested directory structure
def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


def untar(fname, dirs):
    try:
        t = tarfile.open(fname)
        t.extractall(path=dirs)
        return True
    except Exception as e:
        print(e)
        return False


deribit_details = get_exchange_details("binance")

datasets.download(
    exchange="binance",
    data_types=[
        args.data_types,
    ],
    from_date=args.start_date,
    to_date=args.end_date,
    symbols=[
        args.symbols,
    ],
    # (optional) your API key to get access to non sample data as well
    api_key="You should replace this str with your own API key, bought from https://tardis.dev/#pricing, \
    the experiment for EarnHFT is downloaded using the spot data (https://arxiv.org/pdf/2309.12891.pdf)",
    download_dir="./download_from_tardis/{}/{}".format(args.symbols, args.data_types),
)
gz_file_list = os.listdir(
    "./download_from_tardis/{}/{}".format(args.symbols, args.data_types)
)
gz_file_list = [i for i in gz_file_list if i.endswith(".gz")]
for gz_file in gz_file_list:
    with gzip.open(
        os.path.join(
            "./download_from_tardis/{}/{}".format(args.symbols, args.data_types),
            gz_file,
        ),
        "rb",
    ) as f_in:
        with open(
            os.path.join(
                "./download_from_tardis/{}/{}".format(args.symbols, args.data_types),
                gz_file[:-3],
            ),
            "wb",
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(
        os.path.join(
            "./download_from_tardis/{}/{}".format(args.symbols, args.data_types),
            gz_file,
        )
    )
