nohup python ic_analysis/calculate_ic.py \
    --data_path preprocess/merge/BTCTUSD/2023-03-30-2023-05-15/df.feather \
    >ic_analysis/log/TUSD_merge.log 2>&1 &


nohup python ic_analysis/calculate_ic.py \
    --data_path preprocess/merge/BTCUSDT/2022-09-01-2022-10-15/df.feather \
    >ic_analysis/log/USDT_merge.log 2>&1 &


nohup python ic_analysis/calculate_ic.py \
    --data_path preprocess/merge/ETHUSDT/2022-05-01-2022-06-15/df.feather \
    >ic_analysis/log/ETHUSDT_merge.log 2>&1 &


nohup python ic_analysis/calculate_ic.py \
    --data_path preprocess/merge/GALAUSDT/2022-07-01-2022-08-15/df.feather \
    >ic_analysis/log/GALAUSDT_merge.log 2>&1 &