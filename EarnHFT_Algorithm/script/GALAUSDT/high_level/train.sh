CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/high_level/dqn_position.py \
    --train_data_path data/GALAUSDT/train.feather --dataset_name GALAUSDT --max_holding_number 4000  \
    >log/train/GALAUSDT/high_level/log.log 2>&1 &
