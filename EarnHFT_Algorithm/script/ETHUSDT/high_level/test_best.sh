nohup python RL/agent/high_level/test_dqn_position_micro_action.py \
    --test_path result_risk/ETHUSDT/high_level/seed_12345/epoch_58 --dataset_name ETHUSDT --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather --max_holding_number 0.1 \
    >log/test/ETHUSDT/test_best.log 2>&1 &
