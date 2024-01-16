CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/high_level/dqn.py \
    --train_data_path data/GALAUSDT/train.feather --dataset_name GALAUSDT --max_holding_number 4000   \
    >log/train/GALAUSDT/high_level_without_position/log.log 2>&1 &
