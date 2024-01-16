nohup python preprocess/create_feature.py \
--data_path preprocess/concat_clean/BTCTUSD/2023-03-30-2023-05-15 \
>preprocess/log/BTCTUSD_create.log 2>&1 &


nohup python preprocess/create_feature.py \
--data_path preprocess/concat_clean/BTCUSDT/2022-09-01-2022-10-15 \
>preprocess/log/BTCUSDT_create.log 2>&1 &


nohup python preprocess/create_feature.py \
--data_path preprocess/concat_clean/ETHUSDT/2022-05-01-2022-06-15 \
>preprocess/log/ETHUSDT_create.log 2>&1 &


nohup python preprocess/create_feature.py \
--data_path preprocess/concat_clean/GALAUSDT/2022-07-01-2022-08-15 \
>preprocess/log/GALAUSDT_create.log 2>&1 &




nohup python preprocess/create_feature.py \
--data_path preprocess/concat_clean/BTCUSDT/2023-06-19-2023-10-15 \
>preprocess/log/GALAUSDT_create.log 2>&1 &

nohup python preprocess/create_feature.py \
--data_path preprocess/concat_clean/BTCUSDT/2023-10-08-2023-10-18 \
>preprocess/log/GALAUSDT_create.log 2>&1 &


nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-10-08-2023-10-18 --beat_fee 1e-5 \
>preprocess/log/GALAUSDT_create.log 2>&1 &





nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-10-18-2023-10-30 --beat_fee 1e-5 \
>preprocess/log/GALAUSDT_create.log 2>&1 &


nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-09-01-2023-09-30 --beat_fee 1e-5 \
>preprocess/log/BTCUSDT_create.log 2>&1 &

nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-10-01-2023-10-30 --beat_fee 1e-5 \
>preprocess/log/GALAUSDT_create.log 2>&1 &



nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-08-01-2023-08-31 --beat_fee 1e-5 \
>preprocess/log/BTCUSDT_08.log 2>&1 &


nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-07-01-2023-07-31 --beat_fee 1e-5 \
>preprocess/log/BTCUSDT_07.log 2>&1 &





nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-10-31-2023-11-30 --beat_fee 1e-5 \
>preprocess/log/BTCUSDT_07.log 2>&1 &


nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2022-11-30-2022-12-31 --beat_fee 1e-4 \
>preprocess/log/BTCUSDT_create.log 2>&1 &

nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2022-10-31-2022-11-30 --beat_fee 1e-4 \
>preprocess/log/GALAUSDT_create.log 2>&1 &




nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-01-01-2023-01-31 --beat_fee 1e-4 \
>preprocess/log/BTCUSDT_create_01_c.log 2>&1 &

nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-02-01-2023-02-28 --beat_fee 1e-4 \
>preprocess/log/BTCUSDT_create_02_c.log 2>&1 &

nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-03-01-2023-03-31 --beat_fee 1e-4 \
>preprocess/log/BTCUSDT_create_03_c.log 2>&1 &

nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-04-01-2023-04-30 --beat_fee 1e-4 \
>preprocess/log/BTCUSDT_create_04_c.log 2>&1 &


nohup python preprocess/create_feature.py \
    --data_path preprocess/concat_clean/BTCUSDT/2023-05-01-2023-05-31 --beat_fee 1e-4 \
>preprocess/log/BTCUSDT_create_05_c.log 2>&1 &
