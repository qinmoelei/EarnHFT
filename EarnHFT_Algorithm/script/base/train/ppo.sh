CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/base/ppo_train.py \
    --dataset_name BTCTUSD --train_data_path data/BTCTUSD/train --transcation_cost 0 --max_holding_number 0.01 \
    >log/base/BTCTUSD/train_ppo.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/base/ppo_train.py \
    --dataset_name BTCUSDT --train_data_path data/BTCUSDT/train --transcation_cost 0.00015 --max_holding_number 0.01 \
    >log/base/BTCUSDT/train_ppo.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/base/ppo_train.py \
    --dataset_name ETHUSDT --train_data_path data/ETHUSDT/train --transcation_cost 0.00015 --max_holding_number 0.1 \
    >log/base/ETHUSDT/train_ppo.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/base/ppo_train.py \
    --dataset_name GALAUSDT --train_data_path data/GALAUSDT/train --transcation_cost 0.00015 --max_holding_number 4000 \
    >log/base/GALAUSDT/train_ppo.log 2>&1 &
