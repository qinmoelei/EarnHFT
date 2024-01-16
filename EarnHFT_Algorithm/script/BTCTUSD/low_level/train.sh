CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta 100 --train_data_path  data/BTCTUSD/train --dataset_name BTCTUSD  --transcation_cost 0 \
    >log/train/BTCTUSD/low_level/beta_100.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta -10 --train_data_path  data/BTCTUSD/train --dataset_name BTCTUSD --transcation_cost 0 \
    >log/train/BTCTUSD/low_level/beta_-10.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta -90 --train_data_path  data/BTCTUSD/train --dataset_name BTCTUSD --transcation_cost 0 \
    >log/train/BTCTUSD/low_level/beta_-90.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/low_level/ddqn_pes_risk_aware.py \
    --beta 30 --train_data_path  data/BTCTUSD/train --dataset_name BTCTUSD --transcation_cost 0 \
    >log/train/BTCTUSD/low_level/beta_30.log 2>&1 &



