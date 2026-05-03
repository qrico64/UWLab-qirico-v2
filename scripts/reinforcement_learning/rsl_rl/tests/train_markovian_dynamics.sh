#!/bin/bash -l
cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2
source a.sh

python scripts/reinforcement_learning/rsl_rl/tests/train_markovian_dynamics.py \
  --data collected_data/data_may2_r3n1_2/trajectories.pkl \
  --obs-key policy \
  --with-dynamic-parameters \
  --hidden-dims 512,512,512 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 4096 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --wandb-project markovian_policy \
  --wandb-run-name "may2-markovian_dynamics_r3n1_2_policy_with_dynamics_frictionsplit" \
  --wandb-mode online
