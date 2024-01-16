nohup python RL/agent/high_level/test_dqn_position_micro_action.py \
    --test_path result_risk/BTCTUSD/high_level/seed_12345/epoch_54 --dataset_name BTCTUSD --transcation_cost 0 \
    --valid_data_path data/BTCTUSD/valid.feather --test_data_path data/BTCTUSD/test.feather --max_holding_number 0.1 \
    >log/test/BTCTUSD/test_best.log 2>&1 &