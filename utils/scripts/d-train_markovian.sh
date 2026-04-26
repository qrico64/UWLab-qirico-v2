#!/bin/bash -l
set -euo pipefail

cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2
source a.sh

python scripts/reinforcement_learning/rsl_rl/train_markovian.py \
  --train-data collected_data/data_apr25_a2r2o0015n20_1/trajectories.pkl \
  --test-data collected_data/data_apr25_a2r2o0015n100_2/trajectories.pkl \
  --obs-key policy \
  --output-dir experiments/apr25/markovian_a2r2o0015_policy_n10 \
  --experiment-name markovian_policy \
  --run-name "apr25-markovian_a2r2o0015_policy_n10" \
  --wandb-mode online \
  --seed 42 \
  --epochs 100 \
  --batch-size 4096 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --hidden-dim 512 \
  --dropout 0.3 \
  --train-fraction 0.8 \
  --with_noise_value true \
  --num_train_scenarios 10 \
  --num-workers 4 \
  --device cuda \
  --save-every 10
