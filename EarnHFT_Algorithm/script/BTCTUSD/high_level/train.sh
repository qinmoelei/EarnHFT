CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/high_level/dqn_position.py \
    --train_data_path data/BTCTUSD/train.feather --dataset_name BTCTUSD  --transcation_cost 0 \
    >log/train/BTCTUSD/high_level/log.log 2>&1 &