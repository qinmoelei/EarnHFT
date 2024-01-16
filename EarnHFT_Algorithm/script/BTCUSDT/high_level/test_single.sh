nohup python RL/agent/high_level/test_single_agent_micro_action.py \
    --transcation_cost 0.00015 --max_holding_number 0.01 --dataset_name BTCUSDT \
    --test_data_path data/BTCUSDT/test.feather --valid_data_path data/BTCUSDT/valid.feather \
    --action 0 --save_path result_risk/BTCUSDT/high_level_single_agent \
    >log/test/BTCUSDT/single_agent/log_0.log 2>&1 & 

nohup python RL/agent/high_level/test_single_agent_micro_action.py \
    --transcation_cost 0.00015 --max_holding_number 0.01 --dataset_name BTCUSDT \
    --test_data_path data/BTCUSDT/test.feather --valid_data_path data/BTCUSDT/valid.feather \
    --action 1 --save_path result_risk/BTCUSDT/high_level_single_agent \
    >log/test/BTCUSDT/single_agent/log_1.log 2>&1 & 

nohup python RL/agent/high_level/test_single_agent_micro_action.py \
    --transcation_cost 0.00015 --max_holding_number 0.01 --dataset_name BTCUSDT \
    --test_data_path data/BTCUSDT/test.feather --valid_data_path data/BTCUSDT/valid.feather \
    --action 2 --save_path result_risk/BTCUSDT/high_level_single_agent \
    >log/test/BTCUSDT/single_agent/log_2.log 2>&1 & 

nohup python RL/agent/high_level/test_single_agent_micro_action.py \
    --transcation_cost 0.00015 --max_holding_number 0.01 --dataset_name BTCUSDT \
    --test_data_path data/BTCUSDT/test.feather --valid_data_path data/BTCUSDT/valid.feather \
    --action 3 --save_path result_risk/BTCUSDT/high_level_single_agent \
    >log/test/BTCUSDT/single_agent/log_3.log 2>&1 & 

nohup python RL/agent/high_level/test_single_agent_micro_action.py \
    --transcation_cost 0.00015 --max_holding_number 0.01 --dataset_name BTCUSDT \
    --test_data_path data/BTCUSDT/test.feather --valid_data_path data/BTCUSDT/valid.feather \
    --action 4 --save_path result_risk/BTCUSDT/high_level_single_agent \
    >log/test/BTCUSDT/single_agent/log_4.log 2>&1 & 