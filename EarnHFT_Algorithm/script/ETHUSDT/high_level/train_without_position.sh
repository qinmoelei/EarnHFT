CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/high_level/dqn.py \
    --train_data_path data/ETHUSDT/train.feather --dataset_name ETHUSDT --max_holding_number 0.1   \
    >log/train/ETHUSDT/high_level_without_position/log.log 2>&1 &
