nohup python RL/agent/high_level/test_dqn_position_micro_action.py \
    --test_path result_risk/GALAUSDT/high_level/seed_12345/epoch_27 --dataset_name GALAUSDT --valid_data_path data/GALAUSDT/valid.feather --test_data_path data/GALAUSDT/test.feather --max_holding_number 4000 \
    >log/test/GALAUSDT/test_best.log 2>&1 &
