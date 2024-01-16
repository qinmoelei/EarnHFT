CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_6.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_6 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_6_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_6.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_6 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_6_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_6.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_6 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_6_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_6.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_6 \
    --ada_init 256 \
    >log/abliation/df_6_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_9.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_9 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_9_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_9.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_9 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_9_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_9.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_9 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_9_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_9.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_9 \
    --ada_init 256 \
    >log/abliation/df_9_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_12.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_12 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_12_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_12.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_12 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_12_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_12.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_12 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_12_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_12.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_12 \
    --ada_init 256 \
    >log/abliation/df_12_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_15.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_15 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_15_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_15.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_15 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_15_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_15.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_15 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_15_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_15.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_15 \
    --ada_init 256 \
    >log/abliation/df_15_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_18.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_18 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_18_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_18.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_18 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_18_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_18.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_18 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_18_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_18.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_18 \
    --ada_init 256 \
    >log/abliation/df_18_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_21.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_21 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_21_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_21.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_21 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_21_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_21.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_21 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_21_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_21.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_21 \
    --ada_init 256 \
    >log/abliation/df_21_ada_256_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_24.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_24 \
    --ada_init 0 --ada_min 0 --perfect_trans True \
    >log/abliation/df_24_ada_0_trans_true.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_24.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_24 \
    --ada_init 256 --perfect_trans True \
    >log/abliation/df_24_ada_256_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_24.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_24 \
    --ada_init 0 --ada_min 0 \
    >log/abliation/df_24_ada_0_trans_none.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/agent/abliation/dqn.py \
    --train_data_path data/ETHUSDT/train/df_24.feather --max_holding_number 0.1 --dataset_name ETHUSDT_df_24 \
    --ada_init 256 \
    >log/abliation/df_24_ada_256_trans_true.log 2>&1 &
