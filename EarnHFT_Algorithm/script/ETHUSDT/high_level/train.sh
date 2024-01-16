CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/high_level/dqn_position.py \
    --train_data_path data/ETHUSDT/train.feather --dataset_name ETHUSDT --max_holding_number 0.1  \
    >log/train/ETHUSDT/high_level/log.log 2>&1 &