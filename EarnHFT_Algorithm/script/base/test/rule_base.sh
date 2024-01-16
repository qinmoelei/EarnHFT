# #MACD tuning the long term, mid term and short term para
# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 26 --mid_term 12 --short_term 9 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_26_12_9.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 26 --mid_term 12 --short_term 9 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_26_12_9.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 26 --mid_term 12 --short_term 9 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_26_12_9.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 26 --mid_term 12 --short_term 9 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_26_12_9.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 3600 --mid_term 1200 --short_term 60 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_3600_1200_60.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 3600 --mid_term 1200 --short_term 60 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_3600_1200_60.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 3600 --mid_term 1200 --short_term 60 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_3600_1200_60.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 3600 --mid_term 1200 --short_term 60 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_3600_1200_60.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 1200 --mid_term 600 --short_term 60 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_1200_600_60.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 1200 --mid_term 600 --short_term 60 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_1200_600_60.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 1200 --mid_term 600 --short_term 60 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_1200_600_60.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --long_term 1200 --mid_term 600 --short_term 60 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_1200_600_60.log 2>&1 &


# #Imbalance_Volume
# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.1 --lower_theshold -0.1 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_iv_01_01.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.1 --lower_theshold -0.1 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_iv_01_01.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.1 --lower_theshold -0.1 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_iv_01_01.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.1 --lower_theshold -0.1 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_iv_01_01.log 2>&1 &





# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.2 --lower_theshold -0.2 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_iv_02_02.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.2 --lower_theshold -0.2 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_iv_02_02.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.2 --lower_theshold -0.2 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_iv_02_02.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.2 --lower_theshold -0.2 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_iv_02_02.log 2>&1 &


# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.25 --lower_theshold -0.25 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_iv_25_25.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.25 --lower_theshold -0.25 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_iv_25_25.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.25 --lower_theshold -0.25 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_iv_25_25.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.25 --lower_theshold -0.25 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_iv_25_25.log 2>&1 &



# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.5 --lower_theshold -0.5 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_iv_50_50.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.5 --lower_theshold -0.5 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_iv_50_50.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.5 --lower_theshold -0.5 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_iv_50_50.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.5 --lower_theshold -0.5 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_iv_50_50.log 2>&1 &


# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.7 --lower_theshold -0.7 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_iv_70_70.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.7 --lower_theshold -0.7 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_iv_70_70.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.7 --lower_theshold -0.7 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_iv_70_70.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.7 --lower_theshold -0.7 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_iv_70_70.log 2>&1 &


# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.95 --lower_theshold -0.95 --dataset_name BTCTUSD \
#     --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
#     --transcation_cost 0 --max_holding_number 0.01 \
#     >log/base/BTCTUSD/train_macd_iv_95_95.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.95 --lower_theshold -0.95 --dataset_name BTCUSDT \
#     --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.01 \
#     >log/base/BTCUSDT/train_macd_iv_95_95.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.95 --lower_theshold -0.95 --dataset_name ETHUSDT \
#     --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 0.1 \
#     >log/base/ETHUSDT/train_macd_iv_95_95.log 2>&1 &

# nohup python RL/agent/base/rule_based_tune.py \
#     --technicail_indicator Imbalance_Volume --upper_theshold 0.95 --lower_theshold -0.95 --dataset_name GALAUSDT \
#     --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
#     --transcation_cost 0.00015 --max_holding_number 4000 \
#     >log/base/GALAUSDT/train_macd_iv_95_95.log 2>&1 &


nohup python RL/agent/base/rule_based_tune.py \
    --technicail_indicator Imbalance_Volume --upper_theshold 0.99 --lower_theshold -0.99 --dataset_name BTCTUSD \
    --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather \
    --transcation_cost 0 --max_holding_number 0.01 \
    >log/base/BTCTUSD/train_macd_iv_99_99.log 2>&1 &

nohup python RL/agent/base/rule_based_tune.py \
    --technicail_indicator Imbalance_Volume --upper_theshold 0.99 --lower_theshold -0.99 --dataset_name BTCUSDT \
    --valid_data_path data/BTCUSDT/valid.feather --test_data_path data/BTCUSDT/test.feather \
    --transcation_cost 0.00015 --max_holding_number 0.01 \
    >log/base/BTCUSDT/train_macd_iv_99_99.log 2>&1 &

nohup python RL/agent/base/rule_based_tune.py \
    --technicail_indicator Imbalance_Volume --upper_theshold 0.99 --lower_theshold -0.99 --dataset_name ETHUSDT \
    --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather \
    --transcation_cost 0.00015 --max_holding_number 0.1 \
    >log/base/ETHUSDT/train_macd_iv_99_99.log 2>&1 &

nohup python RL/agent/base/rule_based_tune.py \
    --technicail_indicator Imbalance_Volume --upper_theshold 0.99 --lower_theshold -0.99 --dataset_name GALAUSDT \
    --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather \
    --transcation_cost 0.00015 --max_holding_number 4000 \
    >log/base/GALAUSDT/train_macd_iv_99_99.log 2>&1 &