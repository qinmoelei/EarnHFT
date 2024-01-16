CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/high_level/dqn.py \
    --train_data_path data/BTCUSDT/train.feather --dataset_name BTCUSDT   \
    >log/train/BTCUSDT/high_level_without_position/log.log 2>&1 &
