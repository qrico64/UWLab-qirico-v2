#!/bin/bash -l
set -euo pipefail

cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2
source a.sh

python scripts/reinforcement_learning/rsl_rl/train_trajectory_conditioned.py \
  --train-data collected_data/data_a2r2o0015n20_3/trajectories.pkl \
  --test-data collected_data/data_a2r2o0015n100_2/trajectories.pkl \
  --obs-key policy \
  --output-dir experiments/apr20/transformer_a2r2o0015_policy_ntrain10_moredata \
  --experiment-name markovian_policy \
  --run-name "apr20-transformer_a2r2o0015_policy_ntrain10_moredata" \
  --wandb-mode online \
  --seed 42 \
  --epochs 80 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --hidden-dim 256 \
  --num-layers 4 \
  --num-heads 8 \
  --action-head-hidden-dim 256 \
  --dropout 0.1 \
  --train-fraction 0.8 \
  --with_noise_value false \
  --num_train_scenarios 10 \
  --context-length 60 \
  --grad-clip-norm 1.0 \
  --num-workers 4 \
  --device cuda \
  --save-every 20
