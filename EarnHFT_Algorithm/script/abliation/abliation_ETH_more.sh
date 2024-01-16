CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_3.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_3 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_3_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_3.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_3 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_3_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_3.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_3 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_3_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_3.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_3 \
    --ada_init 256 \
    >log/abliation/df_3_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_27.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_27 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_27_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_27.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_27 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_27_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_27.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_27 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_27_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_27.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_27 \
    --ada_init 256 \
    >log/abliation/df_27_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_30.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_30 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_30_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_30.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_30 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_30_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_30.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_30 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_30_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_30.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_30 \
    --ada_init 256 \
    >log/abliation/df_30_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_33.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_33 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_33_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_33.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_33 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_33_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_33.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_33 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_33_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_33.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_33 \
    --ada_init 256 \
    >log/abliation/df_33_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_36.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_36 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_36_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_36.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_36 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_36_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_36.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_36 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_36_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_36.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_36 \
    --ada_init 256 \
    >log/abliation/df_36_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_39.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_39 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_39_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_39.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_39 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_39_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_39.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_39 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_39_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_39.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_39 \
    --ada_init 256 \
    >log/abliation/df_39_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_42.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_42 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_42_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_42.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_42 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_42_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_42.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_42 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_42_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_42.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_42 \
    --ada_init 256 \
    >log/abliation/df_42_ada_256_trans_true.log 2>&1 &
