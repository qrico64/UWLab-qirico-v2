#!/bin/bash -l
cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2
source a.sh

python scripts/reinforcement_learning/rsl_rl/tests/train_markovian_retrieval_wm_noises.py \
  --data collected_data/data_may6_r4n2k_per50_recxgeq05/trajectories.pkl \
  --test-data collected_data/data_may6_r4n2k_per50_xleq035/trajectories.pkl \
  --train-noise-fraction 1 \
  --val-fraction 0.2 \
  --trajectories-per-noise 50 \
  --expected-horizon 60 \
  --expected-num-noises 2000 \
  --min-trajectory-length 10 \
  --max-action-magnitude 100.0 \
  --hidden-dims 2048,2048,2048 \
  --dropout 0.3 \
  --epochs 100 \
  --batch-size 4096 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --retrieval-mode state_action \
  --retrieval-action-multiplier 30 \
  --save_path experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_sa_x30 \
  --wandb-project markovian_policy \
  --wandb-run-name "may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_sa_x30" \
  --wandb-mode online
